from __future__ import annotations

import torch


def simple_atlas_loss(prediction: torch.Tensor, target_rgb: torch.Tensor) -> dict[str, torch.Tensor]:
    """Plain RGB atlas loss: predicted atlas PNG vs expected atlas PNG."""
    pred_rgb = prediction.sigmoid()
    rgb = (pred_rgb - target_rgb).abs().mean()
    return {"total": rgb, "rgb": rgb}
