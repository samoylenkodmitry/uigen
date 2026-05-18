from __future__ import annotations

import torch
import torch.nn.functional as F

from atlas_ai.export_spec import TRAINABLE_EXPORT_SPECS, ExportFileSpec, crop_export_target
from atlas_ai.support_mask import load_support_masks


def sobel_edges(rgb: torch.Tensor) -> torch.Tensor:
    channels = rgb.shape[1]
    kernel_x = rgb.new_tensor(
        [
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0],
        ]
    ) / 4.0
    kernel_y = rgb.new_tensor(
        [
            [-1.0, -2.0, -1.0],
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 1.0],
        ]
    ) / 4.0
    kernel = torch.stack((kernel_x, kernel_y), dim=0)
    kernel = kernel[None, :, None, :, :].expand(channels, 2, 1, 3, 3).reshape(2 * channels, 1, 3, 3)
    return F.conv2d(rgb, kernel, padding=1, groups=channels)


def sobel_l1(pred_rgb: torch.Tensor, target_rgb: torch.Tensor) -> torch.Tensor:
    return (sobel_edges(pred_rgb) - sobel_edges(target_rgb)).abs().mean()


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over True mask positions, averaged across batch and channel dims."""
    pixels = mask.sum().clamp(min=1).to(values.dtype)
    batch = float(values.shape[0])
    channels = float(values.shape[1])
    denom = pixels * batch * channels
    return (values * mask).sum() / denom


def exported_files_loss(
    file_logits: dict[str, torch.Tensor],
    target_rgb: torch.Tensor,
    *,
    specs: tuple[ExportFileSpec, ...] = TRAINABLE_EXPORT_SPECS,
    edge_weight: float = 1.5,
) -> dict[str, torch.Tensor]:
    """Loss over exact exported BMP pixels, normalized per trainable file.

    Pixels Cranamp never reads (per `configs/supported_pixels_classic.json`)
    are excluded from both the loss and the per-file metrics, so the model
    is only graded on what it actually has to reproduce.
    """
    total = target_rgb.new_tensor(0.0)
    weighted_l1 = target_rgb.new_tensor(0.0)
    weighted_edge = target_rgb.new_tensor(0.0)
    weighted_hit5 = target_rgb.new_tensor(0.0)
    unweighted_l1 = target_rgb.new_tensor(0.0)
    unweighted_edge = target_rgb.new_tensor(0.0)
    unweighted_hit5 = target_rgb.new_tensor(0.0)
    total_weight = 0.0
    metrics: dict[str, torch.Tensor] = {}

    support_masks = load_support_masks()

    for spec in specs:
        if spec.file_name not in file_logits:
            raise KeyError(f"missing prediction for {spec.file_name}")
        pred_rgb = file_logits[spec.file_name].sigmoid()
        target_crop = crop_export_target(target_rgb, spec)
        if pred_rgb.shape != target_crop.shape:
            raise ValueError(
                f"{spec.file_name} prediction shape {tuple(pred_rgb.shape)} "
                f"does not match target crop {tuple(target_crop.shape)}"
            )
        # Custom synthetic specs (used in tests) may not appear in the static
        # support profile; default to "every pixel supported" for them.
        cached_mask = support_masks.get(spec.file_name)
        if cached_mask is not None and cached_mask.shape == target_crop.shape[-2:]:
            mask = cached_mask.to(device=target_crop.device)
        else:
            mask = torch.ones(target_crop.shape[-2:], dtype=torch.bool, device=target_crop.device)
        mask_f = mask.to(target_crop.dtype)

        diff = (pred_rgb - target_crop).abs()
        l1 = _masked_mean(diff, mask_f)

        # Substitute the target inside masked-out pixels so Sobel gradients on
        # supported pixels do not leak from unsupported neighbors.
        pred_for_edges = pred_rgb * mask_f + target_crop.detach() * (1 - mask_f)
        edge_diff = (sobel_edges(pred_for_edges) - sobel_edges(target_crop)).abs()
        edge = _masked_mean(edge_diff, mask_f)

        file_loss = l1 + edge_weight * edge
        weight = float(spec.weight)
        total = total + weight * file_loss
        weighted_l1 = weighted_l1 + weight * l1
        weighted_edge = weighted_edge + weight * edge

        hit5_per_chan = (diff <= (5.0 / 255.0)).to(target_crop.dtype)
        hit5 = _masked_mean(hit5_per_chan, mask_f)
        weighted_hit5 = weighted_hit5 + weight * hit5

        unweighted_l1 = unweighted_l1 + l1
        unweighted_edge = unweighted_edge + edge
        unweighted_hit5 = unweighted_hit5 + hit5
        total_weight += weight

        stem = spec.file_name.lower().removesuffix(".bmp")
        metrics[f"mae_{stem}"] = l1
        metrics[f"sobel_{stem}"] = edge

    denom = target_rgb.new_tensor(total_weight)
    file_count = target_rgb.new_tensor(float(len(specs)))
    metrics["total"] = total / denom
    metrics["exported_l1"] = unweighted_l1 / file_count
    metrics["exported_sobel"] = unweighted_edge / file_count
    metrics["exported_hit5"] = unweighted_hit5 / file_count
    metrics["weighted_exported_l1"] = weighted_l1 / denom
    metrics["weighted_exported_sobel"] = weighted_edge / denom
    metrics["weighted_exported_hit5"] = weighted_hit5 / denom
    return metrics
