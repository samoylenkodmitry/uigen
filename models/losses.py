from __future__ import annotations

import torch
import torch.nn.functional as F


def mask_normalized_mean(values: torch.Tensor, mask: torch.Tensor, channel_multiplier: int = 1) -> torch.Tensor:
    denom = mask.sum().clamp_min(1e-8) * channel_multiplier
    return (values * mask).sum() / denom


def rgb_l1_loss(pred_rgb: torch.Tensor, target_rgb: torch.Tensor, effective_mask: torch.Tensor) -> torch.Tensor:
    return mask_normalized_mean((pred_rgb - target_rgb).abs(), effective_mask, channel_multiplier=pred_rgb.shape[1])


def sobel_edges(rgb: torch.Tensor) -> torch.Tensor:
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=rgb.device,
        dtype=rgb.dtype,
    )
    kernel_y = kernel_x.t()
    weight_x = kernel_x.view(1, 1, 3, 3).repeat(rgb.shape[1], 1, 1, 1)
    weight_y = kernel_y.view(1, 1, 3, 3).repeat(rgb.shape[1], 1, 1, 1)
    gx = F.conv2d(rgb, weight_x, padding=1, groups=rgb.shape[1])
    gy = F.conv2d(rgb, weight_y, padding=1, groups=rgb.shape[1])
    return torch.cat((gx, gy), dim=1)


def sobel_l1_loss(pred_rgb: torch.Tensor, target_rgb: torch.Tensor, effective_mask: torch.Tensor) -> torch.Tensor:
    pred_edges = sobel_edges(pred_rgb)
    target_edges = sobel_edges(target_rgb)
    edge_mask = effective_mask.repeat(1, pred_edges.shape[1], 1, 1)
    return mask_normalized_mean((pred_edges - target_edges).abs(), edge_mask, channel_multiplier=1)


def special_ce_loss(logits: torch.Tensor, target: torch.Tensor, atlas_mask: torch.Tensor, enabled: bool) -> torch.Tensor:
    if not enabled:
        return logits.sum() * 0.0
    active = atlas_mask[:, 0] > 0.5
    if not active.any():
        return logits.sum() * 0.0
    loss = F.cross_entropy(logits, target.long(), reduction="none")
    return loss[active].mean()


def slot_loss(
    prediction: torch.Tensor,
    target_rgb: torch.Tensor,
    effective_mask: torch.Tensor,
    atlas_mask: torch.Tensor,
    special_target: torch.Tensor | None = None,
    special_enabled: bool = False,
    slot_weight: torch.Tensor | float = 1.0,
    sobel_weight: float = 0.5,
) -> dict[str, torch.Tensor]:
    pred_rgb = prediction[:, 0:3].sigmoid()
    special_logits = prediction[:, 3:7]
    if special_target is None:
        special_target = torch.zeros(
            prediction.shape[0],
            prediction.shape[2],
            prediction.shape[3],
            device=prediction.device,
            dtype=torch.long,
        )
    l_rgb = rgb_l1_loss(pred_rgb, target_rgb, effective_mask)
    l_sobel = sobel_l1_loss(pred_rgb, target_rgb, effective_mask)
    l_special = special_ce_loss(special_logits, special_target, atlas_mask, special_enabled)
    if not torch.is_tensor(slot_weight):
        slot_weight = torch.tensor(float(slot_weight), device=prediction.device, dtype=prediction.dtype)
    total = slot_weight.mean() * (l_rgb + sobel_weight * l_sobel + l_special)
    return {"total": total, "rgb": l_rgb, "sobel": l_sobel, "special": l_special}


def atlas_contrastive_loss(pred_rgb: torch.Tensor, target_rgb: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Supervised contrastive (InfoNCE) on downsampled atlas embeddings.

    Anti-collapse signal: pred[i] should be closer to its own target[i] than
    to other targets in the batch. But variants of the SAME skin share the
    same atlas target -- treating them as negatives would penalize the model
    for what is actually correct. So we group rows by target_rgb equality
    via torch.unique and treat same-group rows as POSITIVES (supervised
    contrastive). Same-group cells go in the numerator AND the denominator.

    Returns 0 when batch_size < 2 or when all rows share the same group.
    """
    B = pred_rgb.shape[0]
    if B < 2:
        return pred_rgb.sum() * 0.0
    # Group batch rows by per-pixel target equality. Two rows in the same
    # group share the same atlas target (e.g. variants of one skin).
    flat_targets = target_rgb.detach().reshape(B, -1)
    _, group_ids = torch.unique(flat_targets, dim=0, return_inverse=True)
    if int(group_ids.unique().numel()) == 1:
        # Degenerate: all rows are the same skin; nothing to contrast.
        return pred_rgb.sum() * 0.0

    pred = F.interpolate(pred_rgb, size=(64, 64), mode="bilinear", align_corners=False)
    targ = F.interpolate(target_rgb, size=(64, 64), mode="bilinear", align_corners=False)
    pred = F.normalize(pred.flatten(1), dim=1)
    targ = F.normalize(targ.flatten(1), dim=1)
    logits = pred @ targ.t() / temperature  # [B, B]
    positive_mask = (group_ids[:, None] == group_ids[None, :])  # [B, B] bool
    log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    num_pos = positive_mask.sum(dim=1).clamp_min(1)
    # Mean log-prob over positives per anchor.
    masked_lp = log_probs.masked_fill(~positive_mask, 0.0)
    pos_mean = masked_lp.sum(dim=1) / num_pos
    return -pos_mean.mean()


def full_atlas_loss(
    prediction: torch.Tensor,         # [B, 7, H, W] raw logits
    target_rgb: torch.Tensor,         # [B, 3, H, W] in [0, 1]
    atlas_mask: torch.Tensor,         # [B, 1, H, W] in {0, 1}
    effective_mask: torch.Tensor,     # [B, 1, H, W] in [0, 1]
    weight_map: torch.Tensor,         # [B, 1, H, W] per-slot loss weight
    special_target: torch.Tensor,     # [B, H, W] long
    special_mask: torch.Tensor,       # [B, 1, H, W] in {0, 1}
    sobel_weight: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Loss for SlotNetV3 over the full 1024x1024 atlas in one shot.

    All slots are trained simultaneously in a single forward, weighted by the
    per-pixel weight_map (atlas slot loss_weight x per-skin per-slot weight).
    Mask normalization uses (effective_mask * weight_map).sum() so a sparse
    big-area-padded slot contributes the same per-active-pixel gradient as a
    small dense slot.
    """
    pred_rgb = prediction[:, 0:3].sigmoid()
    special_logits = prediction[:, 3:7]

    weighted_mask = effective_mask * weight_map  # [B, 1, H, W]
    denom = weighted_mask.sum().clamp_min(1e-8)

    abs_diff = (pred_rgb - target_rgb).abs()
    l_rgb = (abs_diff * weighted_mask).sum() / (denom * pred_rgb.shape[1])

    pred_edges = sobel_edges(pred_rgb)
    target_edges = sobel_edges(target_rgb)
    edge_mask = weighted_mask.repeat(1, pred_edges.shape[1], 1, 1)
    l_sobel = ((pred_edges - target_edges).abs() * edge_mask).sum() / (
        denom * pred_edges.shape[1] + 1e-8
    )

    # Special CE only where special_mask is active AND atlas_mask is active.
    sp_active = (special_mask[:, 0] > 0.5) & (atlas_mask[:, 0] > 0.5)
    if sp_active.any():
        ce = F.cross_entropy(special_logits, special_target.long(), reduction="none")
        l_special = ce[sp_active].mean()
    else:
        l_special = special_logits.sum() * 0.0

    total = l_rgb + sobel_weight * l_sobel + l_special
    return {"total": total, "rgb": l_rgb, "sobel": l_sobel, "special": l_special}


def full_atlas_loss_v31(
    prediction: torch.Tensor,
    target_rgb: torch.Tensor,
    atlas_mask: torch.Tensor,
    effective_mask: torch.Tensor,
    weight_map: torch.Tensor,
    special_target: torch.Tensor,
    special_mask: torch.Tensor,
    sobel_weight: float = 2.0,
    contrast_weight: float = 0.05,
) -> dict[str, torch.Tensor]:
    """V3.1 loss: full-atlas L1 + Sobel (visible-weighted) + special CE + contrastive."""
    pred_rgb = prediction[:, 0:3].sigmoid()
    special_logits = prediction[:, 3:7]

    weighted_mask = effective_mask * weight_map
    denom = weighted_mask.sum().clamp_min(1e-8)

    abs_diff = (pred_rgb - target_rgb).abs()
    l_rgb = (abs_diff * weighted_mask).sum() / (denom * pred_rgb.shape[1])

    pred_edges = sobel_edges(pred_rgb)
    target_edges = sobel_edges(target_rgb)
    edge_mask = weighted_mask.repeat(1, pred_edges.shape[1], 1, 1)
    l_sobel = ((pred_edges - target_edges).abs() * edge_mask).sum() / (
        denom * pred_edges.shape[1] + 1e-8
    )

    sp_active = (special_mask[:, 0] > 0.5) & (atlas_mask[:, 0] > 0.5)
    if sp_active.any():
        ce = F.cross_entropy(special_logits, special_target.long(), reduction="none")
        l_special = ce[sp_active].mean()
    else:
        l_special = special_logits.sum() * 0.0

    # Skip the contrastive computation entirely when its weight is zero --
    # the per-step torch.unique over a full 1024^2 target tensor is
    # expensive enough to matter over a 20k-step run, and the result is
    # multiplied by zero anyway.
    if contrast_weight > 0:
        l_contrast = atlas_contrastive_loss(pred_rgb, target_rgb)
    else:
        l_contrast = pred_rgb.sum() * 0.0

    total = l_rgb + sobel_weight * l_sobel + l_special + contrast_weight * l_contrast
    return {
        "total": total,
        "rgb": l_rgb,
        "sobel": l_sobel,
        "special": l_special,
        "contrast": l_contrast,
    }


def centernet_focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    positives: torch.Tensor | None = None,
) -> torch.Tensor:
    pred = pred.clamp(1e-6, 1.0 - 1e-6)
    # Positives = floor-cell anchors (passed in via reg_mask), not target.eq(1.0).
    # The Gaussian target rarely hits exactly 1.0 due to floor() discretization,
    # so target.eq(1.0) zero-positives the loss and divides ~5M neg pixels by 1.
    if positives is None:
        positives = target.gt(0.99)
    pos = positives.bool()
    neg = ~pos
    if valid_mask is not None:
        valid = valid_mask.bool()
        pos = pos & valid
        neg = neg & valid
    pos_loss = -((1.0 - pred) ** 2) * pred.log() * pos
    neg_loss = -((1.0 - target) ** 4) * (pred**2) * (1.0 - pred).log() * neg
    num_pos = pos.sum().clamp_min(1)
    return (pos_loss.sum() + neg_loss.sum()) / num_pos
