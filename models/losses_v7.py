"""V7 completer Phase 0 losses and metrics.

Loss path is intentionally minimal for Phase 0:
    L = masked_l1(final_rgb, target_rgb, support_mask)

`support_mask` is the static Cranamp-supported-pixel mask from
`atlas_ai.support_mask.load_support_masks()`. Only pixels Cranamp actually
draws contribute to the loss / metrics; the rest of the exported BMP is
padding the renderer never reads.

Metrics (per-file and aggregate, all on supported pixels):
    supported_mae  mean |final - target| in [0, 1] units
    hit5           fraction of pixels with |final - target| * 255 <= 5
    sobel_mae      mean |sobel(final) - sobel(target)|
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


__all__ = [
    "support_masked_l1_loss",
    "support_masked_hit5",
    "support_masked_sobel_mae",
]
