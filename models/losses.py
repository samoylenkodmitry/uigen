from __future__ import annotations

import torch
import torch.nn.functional as F

from atlas_ai.export_spec import TRAINABLE_EXPORT_SPECS, ExportFileSpec, crop_export_target


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


def exported_files_loss(
    file_logits: dict[str, torch.Tensor],
    target_rgb: torch.Tensor,
    *,
    specs: tuple[ExportFileSpec, ...] = TRAINABLE_EXPORT_SPECS,
    edge_weight: float = 1.5,
) -> dict[str, torch.Tensor]:
    """Loss over exact exported BMP pixels, normalized per trainable file."""
    total = target_rgb.new_tensor(0.0)
    weighted_l1 = target_rgb.new_tensor(0.0)
    weighted_edge = target_rgb.new_tensor(0.0)
    weighted_hit5 = target_rgb.new_tensor(0.0)
    unweighted_l1 = target_rgb.new_tensor(0.0)
    unweighted_edge = target_rgb.new_tensor(0.0)
    unweighted_hit5 = target_rgb.new_tensor(0.0)
    total_weight = 0.0
    metrics: dict[str, torch.Tensor] = {}

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
        l1 = (pred_rgb - target_crop).abs().mean()
        edge = sobel_l1(pred_rgb, target_crop)
        file_loss = l1 + edge_weight * edge
        weight = float(spec.weight)
        total = total + weight * file_loss
        weighted_l1 = weighted_l1 + weight * l1
        weighted_edge = weighted_edge + weight * edge
        hit5 = ((pred_rgb - target_crop).abs() <= (5.0 / 255.0)).float().mean()
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
