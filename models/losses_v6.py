"""V6 source-preserving losses.

Per the V6 plan, training combines four terms (Stage 2 omits residual/fallback):

    L_final_l1     (Stage 4+, with fallback)
    L_final_sobel  (Stage 4+, with fallback)
    L_copy_rgb     supported_visible pixels, copy_refined vs clean target
    L_conf         BCE(copy_conf_logits, visible_mask), class-balanced
    L_uv           SmoothL1(uv_grid, uv_target), visible_mask only
    L_uv_tv        low-weight total variation on uv_grid

Stage 2 weights:

    1.00 * L_copy_rgb
    0.25 * L_conf
    0.10 * L_uv
    0.01 * L_uv_tv

The copy composition uses align_corners=False:

    copy_rgb = grid_sample(input_view, uv_grid.permute(0, 2, 3, 1),
                           align_corners=False)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


# Default pos_weight clamp for class-balanced confidence BCE (V6 plan).
DEFAULT_CONF_POS_WEIGHT_CAP = 10.0


def grid_sample_copy(view: torch.Tensor, uv_grid: torch.Tensor) -> torch.Tensor:
    """Copy from the input view using the V6 grid_sample convention.

    Args:
        view: [B, 3, IH, IW] image in [0, 1].
        uv_grid: [B, 2, H, W] coordinates in [-1, 1] (channel 0 = u/x, 1 = v/y).

    Returns: [B, 3, H, W] copied RGB at the file dimensions.
    """
    if view.dim() != 4 or view.shape[1] != 3:
        raise ValueError(f"view must be [B, 3, H, W], got {tuple(view.shape)}")
    if uv_grid.dim() != 4 or uv_grid.shape[1] != 2:
        raise ValueError(f"uv_grid must be [B, 2, H, W], got {tuple(uv_grid.shape)}")
    grid = uv_grid.permute(0, 2, 3, 1)
    return F.grid_sample(view, grid, mode="bilinear", align_corners=False, padding_mode="zeros")


def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Mean of x at positions where mask>0. Numerically safe when mask is empty.

    Args:
        x: any-shape tensor.
        mask: same shape as x, in {0, 1} (uint8 or float).
        eps: minimum denominator. eps>=1 ensures the empty-mask result is the
            sum (which will already be zero), avoiding NaN.
    """
    if x.shape != mask.shape:
        raise ValueError(f"x shape {tuple(x.shape)} != mask shape {tuple(mask.shape)}")
    denom = mask.sum().clamp_min(eps)
    return (x * mask).sum() / denom


def copy_rgb_l1_loss(
    copy_refined: torch.Tensor,
    target_rgb: torch.Tensor,
    visible_mask: torch.Tensor,
) -> torch.Tensor:
    """L1 between copied / refined RGB and clean target, restricted to visible.

    Args:
        copy_refined: [B, 3, H, W] in [0, 1].
        target_rgb: [B, 3, H, W] in [0, 1].
        visible_mask: [B, 1, H, W] or [B, H, W] in {0, 1}.
    """
    if visible_mask.dim() == 3:
        visible_mask = visible_mask.unsqueeze(1)
    mask3 = visible_mask.expand_as(copy_refined).to(copy_refined.dtype)
    diff = (copy_refined - target_rgb).abs()
    return masked_mean(diff, mask3)


def conf_bce_loss(
    conf_logits: torch.Tensor,
    visible_mask: torch.Tensor,
    pos_weight_cap: float = DEFAULT_CONF_POS_WEIGHT_CAP,
) -> torch.Tensor:
    """Per-pixel BCE on copy_conf_logits against visible_mask, class-balanced.

    Args:
        conf_logits: [B, 1, H, W] raw logits.
        visible_mask: [B, 1, H, W] or [B, H, W] in {0, 1}.
        pos_weight_cap: maximum allowed pos_weight to prevent runaway weights
            when only a few pixels are visible.
    """
    if visible_mask.dim() == 3:
        visible_mask = visible_mask.unsqueeze(1)
    target = visible_mask.to(conf_logits.dtype)
    pos = target.sum().clamp_min(1.0)
    neg = (1.0 - target).sum().clamp_min(1.0)
    pos_weight = torch.clamp(neg / pos, max=pos_weight_cap)
    return F.binary_cross_entropy_with_logits(conf_logits, target, pos_weight=pos_weight)


def uv_smooth_l1_loss(
    uv_pred: torch.Tensor,
    uv_target: torch.Tensor,
    visible_mask: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """Smooth-L1 between predicted UV and target UV at visible source pixels.

    Args:
        uv_pred: [B, 2, H, W] in [-1, 1].
        uv_target: [B, 2, H, W] in [-1, 1].
        visible_mask: [B, 1, H, W] or [B, H, W] in {0, 1}.
        beta: smooth-L1 transition point. Default 0.1 (about 5% of UV range).
    """
    if visible_mask.dim() == 3:
        visible_mask = visible_mask.unsqueeze(1)
    mask2 = visible_mask.expand_as(uv_pred).to(uv_pred.dtype)
    per_pixel = F.smooth_l1_loss(uv_pred, uv_target, beta=beta, reduction="none")
    return masked_mean(per_pixel, mask2)


def uv_smooth_l1_pixel_loss(
    uv_pred: torch.Tensor,
    uv_target: torch.Tensor,
    visible_mask: torch.Tensor,
    view_h: int,
    view_w: int,
    beta_px: float = 2.0,
) -> torch.Tensor:
    """Smooth-L1 between predicted and target UV expressed in input view pixels.

    UV is in [-1, 1] under align_corners=False, so 1 unit = (size / 2) pixels.
    Operating in pixel space makes the loss landscape match the Stage 2 eval
    metric (uv_median_px), so the quadratic-to-linear knee sits at the
    acceptance bar instead of far outside it.

    Args:
        uv_pred: [B, 2, H, W] in [-1, 1].
        uv_target: [B, 2, H, W] in [-1, 1].
        visible_mask: [B, 1, H, W] or [B, H, W] in {0, 1}.
        view_h, view_w: input view dimensions used by grid_sample.
        beta_px: smooth-L1 transition in pixels. Default 2.0 (the Stage 2 bar).
    """
    if visible_mask.dim() == 3:
        visible_mask = visible_mask.unsqueeze(1)
    du = (uv_pred[:, 0:1] - uv_target[:, 0:1]) * (view_w / 2.0)
    dv = (uv_pred[:, 1:2] - uv_target[:, 1:2]) * (view_h / 2.0)
    pixel_diff = torch.cat([du, dv], dim=1)
    zeros = torch.zeros_like(pixel_diff)
    per_pixel = F.smooth_l1_loss(pixel_diff, zeros, beta=beta_px, reduction="none")
    mask2 = visible_mask.expand_as(pixel_diff).to(pixel_diff.dtype)
    return masked_mean(per_pixel, mask2)


def uv_total_variation_loss(uv_pred: torch.Tensor) -> torch.Tensor:
    """Per-pixel UV total variation regularization (encourages smooth UV).

    Args:
        uv_pred: [B, 2, H, W].
    """
    dh = (uv_pred[:, :, 1:, :] - uv_pred[:, :, :-1, :]).abs().mean()
    dw = (uv_pred[:, :, :, 1:] - uv_pred[:, :, :, :-1]).abs().mean()
    return 0.5 * (dh + dw)


@dataclass(frozen=True)
class V6CopyLossWeights:
    """Stage 2 / residual-probe loss weights.

    `residual_l1` defaults to 0 so the original Stage 2 recipe is unchanged
    when no residual head is being trained. Set to 0.02-0.05 to penalize
    unnecessary residual activity in the copy_residual probe.
    """
    copy_rgb: float = 1.00
    conf: float = 0.25
    uv: float = 0.10
    uv_tv: float = 0.01
    residual_l1: float = 0.0


def refine_copy_rgb(copy_rgb: torch.Tensor, residual: torch.Tensor | None,
                    residual_scale: float = 0.25) -> torch.Tensor:
    """Stage 2 copy path: clamp(copy_rgb + scale * tanh(residual), 0, 1).

    When `residual` is None, returns clamp(copy_rgb, 0, 1) unchanged so the
    refined path is equivalent to the original copy when no residual head is
    attached.
    """
    refined = copy_rgb
    if residual is not None:
        refined = refined + residual_scale * torch.tanh(residual)
    return refined.clamp(0.0, 1.0)


def residual_l1_penalty(residual: torch.Tensor, visible_mask: torch.Tensor) -> torch.Tensor:
    """L1 penalty on the residual at visible pixels.

    Encourages the model to leave copy_rgb alone where the copy is already
    correct. Computed only over visible pixels so masked-out regions don't
    contribute spurious zero-target signal.
    """
    if visible_mask.dim() == 3:
        visible_mask = visible_mask.unsqueeze(1)
    mask3 = visible_mask.expand_as(residual).to(residual.dtype)
    return masked_mean(residual.abs(), mask3)


def compute_v6_copy_stage_loss(
    *,
    view: torch.Tensor,
    uv_pred: torch.Tensor,
    conf_logits: torch.Tensor,
    target_rgb: torch.Tensor,
    visible_mask: torch.Tensor,
    uv_target: torch.Tensor,
    weights: V6CopyLossWeights = V6CopyLossWeights(),
    uv_loss_mode: str = "normalized",
    uv_beta_px: float = 2.0,
    residual: torch.Tensor | None = None,
    residual_scale: float = 0.25,
) -> dict[str, torch.Tensor]:
    """Combine the Stage 2 V6 copy-head losses on one file.

    Args:
        uv_loss_mode: 'normalized' (smooth-L1 in [-1, 1] space, beta=0.1) or
            'pixel' (smooth-L1 in input-view pixel space, beta=uv_beta_px).
            Pixel mode matches the Stage 2 eval metric and is recommended when
            the model needs to learn precise UV correspondence.
        uv_beta_px: pixel-space smooth-L1 transition; ignored if mode is
            'normalized'.
        residual: optional [B, 3, H, W] correction added to the copied RGB as
            `clamp(copy_rgb + residual_scale * tanh(residual), 0, 1)`. When
            None, the original Stage 2 copy path is used.
        residual_scale: maximum residual magnitude per pixel. Default 0.25
            matches the V6 plan.

    Returns a dict containing each per-term scalar and a 'total' scalar suitable
    for backward(). Per-term scalars are detached for logging.
    """
    copy_rgb = grid_sample_copy(view, uv_pred)
    copy_refined = refine_copy_rgb(copy_rgb, residual, residual_scale)
    l_copy = copy_rgb_l1_loss(copy_refined, target_rgb, visible_mask)
    l_conf = conf_bce_loss(conf_logits, visible_mask)
    if uv_loss_mode == "pixel":
        _, _, view_h, view_w = view.shape
        l_uv = uv_smooth_l1_pixel_loss(
            uv_pred, uv_target, visible_mask, view_h, view_w, beta_px=uv_beta_px,
        )
    elif uv_loss_mode == "normalized":
        l_uv = uv_smooth_l1_loss(uv_pred, uv_target, visible_mask)
    else:
        raise ValueError(f"unknown uv_loss_mode: {uv_loss_mode!r}")
    l_tv = uv_total_variation_loss(uv_pred)
    if residual is not None:
        l_residual = residual_l1_penalty(residual, visible_mask)
    else:
        l_residual = torch.zeros((), device=view.device, dtype=view.dtype)
    total = (
        weights.copy_rgb * l_copy
        + weights.conf * l_conf
        + weights.uv * l_uv
        + weights.uv_tv * l_tv
        + weights.residual_l1 * l_residual
    )
    return {
        "total": total,
        "copy_rgb": l_copy.detach(),
        "conf": l_conf.detach(),
        "uv": l_uv.detach(),
        "uv_tv": l_tv.detach(),
        "residual_l1": l_residual.detach(),
    }


__all__ = [
    "V6CopyLossWeights",
    "compute_v6_copy_stage_loss",
    "conf_bce_loss",
    "copy_rgb_l1_loss",
    "grid_sample_copy",
    "masked_mean",
    "refine_copy_rgb",
    "residual_l1_penalty",
    "uv_smooth_l1_loss",
    "uv_smooth_l1_pixel_loss",
    "uv_total_variation_loss",
    "DEFAULT_CONF_POS_WEIGHT_CAP",
]
