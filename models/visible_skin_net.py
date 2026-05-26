"""VisibleSkinNet: V8 product-facing exported-BMP predictor.

This is the model interface for the V8 pipeline:

    normalized mockup image -> exact exported BMP tensors

Hidden states are deliberately not the model's primary contract; downstream
`hidden_state_compiler` can synthesize plausible alternates from these visible
assets. The initial implementation is a compact per-file-head network so the
training/eval code can target the product pipeline without reusing the old
hidden-atlas benchmark framing.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from atlas_ai.export_spec import TRAINABLE_EXPORT_SPECS, ExportFileSpec
from models.slotnet_v5 import conv_block


def _head_key(file_name: str) -> str:
    return file_name.lower().removesuffix(".bmp").replace(" ", "_")


def _grid_size(spec: ExportFileSpec, divisor: int) -> tuple[int, int]:
    return max(1, math.ceil(spec.h / divisor)), max(1, math.ceil(spec.w / divisor))


class _VisibleFileHead(nn.Module):
    def __init__(self, spec: ExportFileSpec, style_dim: int, hidden: int, divisor: int):
        super().__init__()
        self.spec = spec
        h0, w0 = _grid_size(spec, divisor)
        self.h0 = h0
        self.w0 = w0
        self.seed = nn.Linear(style_dim, hidden * h0 * w0)
        groups = 8 if hidden % 8 == 0 else 1
        self.decode = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 3, 1),
        )

    def forward(self, style: torch.Tensor) -> torch.Tensor:
        b = style.shape[0]
        x = self.seed(style).reshape(b, -1, self.h0, self.w0)
        logits = self.decode(x)
        if logits.shape[-2:] != (self.spec.h, self.spec.w):
            logits = F.interpolate(logits, size=(self.spec.h, self.spec.w), mode="nearest")
        return torch.sigmoid(logits)


class VisibleSkinNet(nn.Module):
    """Predict exact exported BMP tensors from a normalized mockup."""

    def __init__(
        self,
        *,
        base_channels: int = 32,
        style_dim: int = 256,
        head_channels: int = 96,
        head_divisor: int = 8,
    ):
        super().__init__()
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 6
        self.encoder = nn.Sequential(
            conv_block(3, c1, 8, stride=2),
            conv_block(c1, c2, 8, stride=2),
            conv_block(c2, c3, 16, stride=2),
            conv_block(c3, c4, 16, stride=2),
        )
        self.style = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c4, style_dim),
            nn.SiLU(inplace=True),
            nn.Linear(style_dim, style_dim),
        )
        self.heads = nn.ModuleDict(
            {
                _head_key(spec.file_name): _VisibleFileHead(
                    spec, style_dim=style_dim, hidden=head_channels, divisor=head_divisor
                )
                for spec in TRAINABLE_EXPORT_SPECS
            }
        )

    def forward(self, view: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
        if view.dim() != 4 or view.shape[1] != 3:
            raise ValueError(f"view must be [B,3,H,W], got {tuple(view.shape)}")
        style = self.style(self.encoder(view))
        files = {
            spec.file_name: self.heads[_head_key(spec.file_name)](style)
            for spec in TRAINABLE_EXPORT_SPECS
        }
        return {"files": files}


__all__ = ["VisibleSkinNet"]
