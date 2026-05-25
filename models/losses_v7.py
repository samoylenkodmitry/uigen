"""V7 completer Phase 0 losses and metrics.

Two denominators matter for this task, and they answer different questions.

`support_mask` is the static Cranamp-supported-pixel mask from
`atlas_ai.support_mask.load_support_masks()`. Only pixels Cranamp actually
draws are real; the rest of the exported BMP is padding the renderer never
reads.

Within the support region the completer hard-copies observed pixels:

    final_rgb = observed_mask * observed_rgb + (1 - observed_mask) * generated

so the only pixels the model actually *generates* are the hidden ones:

    support  = support_mask
    observed = observed_mask * support
    hidden   = (1 - observed_mask) * support

`support_masked_*` (the "full_supported" family) normalize by every supported
pixel. Because most of those are hard-copied, mostly-observed samples get a
diluted, optimistic score. They remain useful as a debug/secondary view.

`hidden_supported_*` (the primary train/eval family) normalize by hidden
pixels only, so the metric reflects the part of the problem the model has to
solve. `observed_passthrough_mae` is the diagnostic that the hard copy is
intact (it should be ~0 by construction).

All metrics are restricted to their denominator's pixels and reported in
[0, 1] units (hit5 in fraction-of-pixels).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def support_masked_l1_loss(
    final_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    support_mask: torch.Tensor,
) -> torch.Tensor:
    """L1 between final_rgb and target_rgb restricted to support_mask pixels.

    Args:
        final_rgb:  [B, 3, H, W] in [0, 1].
        target_rgb: [B, 3, H, W] in [0, 1].
        support_mask: [H, W] or [1, H, W] or [B, 1, H, W] in {0, 1}.
    """
    mask = support_mask
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(0)
    if mask.shape[-2:] != final_rgb.shape[-2:]:
        raise ValueError(
            f"support_mask {tuple(mask.shape)} does not match final_rgb {tuple(final_rgb.shape)}"
        )
    mask3 = mask.expand_as(final_rgb).to(final_rgb.dtype)
    diff = (final_rgb - target_rgb).abs() * mask3
    denom = mask3.sum().clamp_min(1.0)
    return diff.sum() / denom


def support_masked_hit5(
    final_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    support_mask: torch.Tensor,
    threshold_levels: float = 5.0,
) -> torch.Tensor:
    """Fraction of supported pixels with all-channel |diff| * 255 <= threshold.

    Returns a scalar in [0, 1]. Matches the V5/V6 hit5 metric definition.
    """
    mask = support_mask
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(0)
    diff = (final_rgb - target_rgb).abs() * 255.0
    # A pixel is a "hit" only if every channel is within the threshold.
    per_pixel_hit = (diff <= threshold_levels).all(dim=1, keepdim=True).to(final_rgb.dtype)
    mask_one = mask.to(final_rgb.dtype).expand_as(per_pixel_hit)
    denom = mask_one.sum().clamp_min(1.0)
    return (per_pixel_hit * mask_one).sum() / denom


def _sobel_kernels(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=device, dtype=dtype,
    )
    ky = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=device, dtype=dtype,
    )
    return kx, ky


def support_masked_sobel_mae(
    final_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    support_mask: torch.Tensor,
) -> torch.Tensor:
    """Sobel-edge MAE on supported pixels."""
    kx, ky = _sobel_kernels(final_rgb.device, final_rgb.dtype)
    weight_x = kx.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    weight_y = ky.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    pred_x = F.conv2d(final_rgb, weight_x, padding=1, groups=3)
    pred_y = F.conv2d(final_rgb, weight_y, padding=1, groups=3)
    tgt_x = F.conv2d(target_rgb, weight_x, padding=1, groups=3)
    tgt_y = F.conv2d(target_rgb, weight_y, padding=1, groups=3)
    pred_mag = (pred_x.pow(2) + pred_y.pow(2)).clamp_min(1e-12).sqrt()
    tgt_mag = (tgt_x.pow(2) + tgt_y.pow(2)).clamp_min(1e-12).sqrt()
    mask = support_mask
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(0)
    mask3 = mask.expand_as(pred_mag).to(pred_mag.dtype)
    diff = (pred_mag - tgt_mag).abs() * mask3
    denom = mask3.sum().clamp_min(1.0)
    return diff.sum() / denom


def support_masked_l1_per_item(
    final_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    support_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-item supported-pixel L1, returned as a [B] tensor.

    Mirrors `support_masked_l1_loss` but reduces only across (C, H, W)
    so the trainer can bucket batch losses by mode / file / skin.
    """
    mask = support_mask
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(0)
    mask3 = mask.expand_as(final_rgb).to(final_rgb.dtype)
    diff = (final_rgb - target_rgb).abs() * mask3
    per_item_num = diff.flatten(1).sum(dim=1)
    per_item_den = mask3.flatten(1).sum(dim=1).clamp_min(1.0)
    return per_item_num / per_item_den


def support_masked_sobel_mae_per_item(
    final_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    support_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-item supported-pixel Sobel-edge MAE, returned as a [B] tensor."""
    kx, ky = _sobel_kernels(final_rgb.device, final_rgb.dtype)
    weight_x = kx.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    weight_y = ky.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    pred_x = F.conv2d(final_rgb, weight_x, padding=1, groups=3)
    pred_y = F.conv2d(final_rgb, weight_y, padding=1, groups=3)
    tgt_x = F.conv2d(target_rgb, weight_x, padding=1, groups=3)
    tgt_y = F.conv2d(target_rgb, weight_y, padding=1, groups=3)
    pred_mag = (pred_x.pow(2) + pred_y.pow(2)).clamp_min(1e-12).sqrt()
    tgt_mag = (tgt_x.pow(2) + tgt_y.pow(2)).clamp_min(1e-12).sqrt()
    mask = support_mask
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(0)
    mask3 = mask.expand_as(pred_mag).to(pred_mag.dtype)
    diff = (pred_mag - tgt_mag).abs() * mask3
    per_item_num = diff.flatten(1).sum(dim=1)
    per_item_den = mask3.flatten(1).sum(dim=1).clamp_min(1.0)
    return per_item_num / per_item_den


# ---------------------------------------------------------------------------
# Hidden-normalized losses and metrics.
#
# The functions below all derive a [B, 1, H, W] weight mask from
# (observed_mask, support_mask) and reduce against it. They return *terms*
# (per-item numerator and denominator) so callers can aggregate correctly:
#   - the training loss wants a single pixel-weighted scalar over the batch
#     (num.sum() / den.sum());
#   - per-item telemetry wants num_i / den_i with hidden-less items excluded;
#   - the eval script wants to accumulate num/den across many batches whose
#     hidden-pixel counts differ.
# Building everything on num/den keeps those three aggregations consistent.
# ---------------------------------------------------------------------------


def _normalize_mask_b1hw(mask: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Coerce a mask to a [*, 1, H, W] tensor broadcastable against ref [B, 3, H, W]."""
    m = mask
    if m.dim() == 2:
        m = m.unsqueeze(0).unsqueeze(0)
    elif m.dim() == 3:
        m = m.unsqueeze(0)
    elif m.dim() != 4:
        raise ValueError(f"mask must be 2-4 dims, got {tuple(mask.shape)}")
    if m.shape[-2:] != ref.shape[-2:]:
        raise ValueError(
            f"mask {tuple(m.shape)} does not match ref {tuple(ref.shape)}"
        )
    return m.to(ref.dtype)


def hidden_support_mask(
    observed_mask: torch.Tensor,
    support_mask: torch.Tensor,
    ref: torch.Tensor,
) -> torch.Tensor:
    """hidden = (1 - observed) * support, as [B, 1, H, W] broadcastable to ref.

    These are exactly the pixels the model must generate: inside the support
    region and not handed to it as observed evidence.
    """
    obs = _normalize_mask_b1hw(observed_mask, ref)
    sup = _normalize_mask_b1hw(support_mask, ref)
    return (1.0 - obs) * sup


def observed_support_mask(
    observed_mask: torch.Tensor,
    support_mask: torch.Tensor,
    ref: torch.Tensor,
) -> torch.Tensor:
    """observed = observed * support, the hard-copied pixels."""
    obs = _normalize_mask_b1hw(observed_mask, ref)
    sup = _normalize_mask_b1hw(support_mask, ref)
    return obs * sup


def _l1_terms(
    final_rgb: torch.Tensor, target_rgb: torch.Tensor, mask_b1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask3 = mask_b1.expand_as(final_rgb).to(final_rgb.dtype)
    diff = (final_rgb - target_rgb).abs() * mask3
    return diff.flatten(1).sum(dim=1), mask3.flatten(1).sum(dim=1)


def _hit5_terms(
    final_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    mask_b1: torch.Tensor,
    threshold_levels: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    diff = (final_rgb - target_rgb).abs() * 255.0
    per_pixel_hit = (diff <= threshold_levels).all(dim=1, keepdim=True).to(final_rgb.dtype)
    m = mask_b1.expand_as(per_pixel_hit).to(final_rgb.dtype)
    return (per_pixel_hit * m).flatten(1).sum(dim=1), m.flatten(1).sum(dim=1)


def _sobel_terms(
    final_rgb: torch.Tensor, target_rgb: torch.Tensor, mask_b1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    kx, ky = _sobel_kernels(final_rgb.device, final_rgb.dtype)
    weight_x = kx.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    weight_y = ky.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    pred_x = F.conv2d(final_rgb, weight_x, padding=1, groups=3)
    pred_y = F.conv2d(final_rgb, weight_y, padding=1, groups=3)
    tgt_x = F.conv2d(target_rgb, weight_x, padding=1, groups=3)
    tgt_y = F.conv2d(target_rgb, weight_y, padding=1, groups=3)
    pred_mag = (pred_x.pow(2) + pred_y.pow(2)).clamp_min(1e-12).sqrt()
    tgt_mag = (tgt_x.pow(2) + tgt_y.pow(2)).clamp_min(1e-12).sqrt()
    mask3 = mask_b1.expand_as(pred_mag).to(pred_mag.dtype)
    diff = (pred_mag - tgt_mag).abs() * mask3
    return diff.flatten(1).sum(dim=1), mask3.flatten(1).sum(dim=1)


def _aggregate(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    """Pixel-weighted batch reduction. Items with den==0 contribute nothing."""
    return num.sum() / den.sum().clamp_min(1.0)


def _per_item(num: torch.Tensor, den: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-item value plus a bool mask of which items had a non-empty denominator."""
    valid = den > 0
    value = torch.where(valid, num / den.clamp_min(1.0), torch.zeros_like(num))
    return value, valid


# -- hidden L1 / MAE --------------------------------------------------------

def hidden_supported_l1_terms(final_rgb, target_rgb, observed_mask, support_mask):
    """Per-item (num, den) for L1 over hidden∩support pixels."""
    hidden = hidden_support_mask(observed_mask, support_mask, final_rgb)
    return _l1_terms(final_rgb, target_rgb, hidden)


def hidden_supported_l1_loss(final_rgb, target_rgb, observed_mask, support_mask):
    """Pixel-weighted L1 over hidden∩support pixels. The primary train loss.

    Observed (hard-copied) pixels are excluded from the denominator, so a
    mostly-observed sample is not diluted into looking near-perfect.
    """
    return _aggregate(*hidden_supported_l1_terms(final_rgb, target_rgb, observed_mask, support_mask))


def hidden_supported_l1_per_item(final_rgb, target_rgb, observed_mask, support_mask):
    """Per-item hidden L1 as [B], plus a [B] bool of items that had hidden pixels.

    Items with no hidden pixels (e.g. passthrough masks) return 0.0 and
    has_hidden=False so callers can drop them from hidden-metric averages.
    """
    return _per_item(*hidden_supported_l1_terms(final_rgb, target_rgb, observed_mask, support_mask))


# -- hidden hit5 ------------------------------------------------------------

def hidden_supported_hit5_terms(final_rgb, target_rgb, observed_mask, support_mask,
                                threshold_levels: float = 5.0):
    hidden = hidden_support_mask(observed_mask, support_mask, final_rgb)
    return _hit5_terms(final_rgb, target_rgb, hidden, threshold_levels)


def hidden_supported_hit5(final_rgb, target_rgb, observed_mask, support_mask,
                          threshold_levels: float = 5.0):
    """Fraction of hidden∩support pixels within `threshold_levels`/255 on all channels."""
    return _aggregate(*hidden_supported_hit5_terms(
        final_rgb, target_rgb, observed_mask, support_mask, threshold_levels))


# -- hidden sobel -----------------------------------------------------------

def hidden_supported_sobel_terms(final_rgb, target_rgb, observed_mask, support_mask):
    hidden = hidden_support_mask(observed_mask, support_mask, final_rgb)
    return _sobel_terms(final_rgb, target_rgb, hidden)


def hidden_supported_sobel_mae(final_rgb, target_rgb, observed_mask, support_mask):
    """Sobel-edge MAE restricted to hidden∩support pixels."""
    return _aggregate(*hidden_supported_sobel_terms(final_rgb, target_rgb, observed_mask, support_mask))


def hidden_supported_sobel_mae_per_item(final_rgb, target_rgb, observed_mask, support_mask):
    return _per_item(*hidden_supported_sobel_terms(final_rgb, target_rgb, observed_mask, support_mask))


# -- observed-passthrough diagnostic ---------------------------------------

def observed_passthrough_terms(final_rgb, target_rgb, observed_mask, support_mask):
    observed = observed_support_mask(observed_mask, support_mask, final_rgb)
    return _l1_terms(final_rgb, target_rgb, observed)


def observed_passthrough_mae(final_rgb, target_rgb, observed_mask, support_mask):
    """MAE over observed∩support pixels. ~0 by construction (the hard copy);
    a non-zero value means the copy-through path is broken."""
    return _aggregate(*observed_passthrough_terms(final_rgb, target_rgb, observed_mask, support_mask))


__all__ = [
    # full-supported family (debug / secondary)
    "support_masked_l1_loss",
    "support_masked_hit5",
    "support_masked_sobel_mae",
    "support_masked_l1_per_item",
    "support_masked_sobel_mae_per_item",
    # hidden-support derivation
    "hidden_support_mask",
    "observed_support_mask",
    # hidden-normalized primary family
    "hidden_supported_l1_terms",
    "hidden_supported_l1_loss",
    "hidden_supported_l1_per_item",
    "hidden_supported_hit5_terms",
    "hidden_supported_hit5",
    "hidden_supported_sobel_terms",
    "hidden_supported_sobel_mae",
    "hidden_supported_sobel_mae_per_item",
    "observed_passthrough_terms",
    "observed_passthrough_mae",
]
