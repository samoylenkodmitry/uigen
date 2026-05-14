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


def centernet_focal_loss(pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
    pred = pred.clamp(1e-6, 1.0 - 1e-6)
    pos = target.eq(1.0)
    neg = target.lt(1.0)
    if valid_mask is not None:
        pos = pos & valid_mask.bool()
        neg = neg & valid_mask.bool()
    pos_loss = -((1.0 - pred) ** 2) * pred.log() * pos
    neg_loss = -((1.0 - target) ** 4) * (pred**2) * (1.0 - pred).log() * neg
    num_pos = pos.sum().clamp_min(1)
    return (pos_loss.sum() + neg_loss.sum()) / num_pos
