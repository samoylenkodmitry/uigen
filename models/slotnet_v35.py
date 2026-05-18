"""SlotNetV3.5: direct image-to-exported-BMP model.

The training contract is:

    input rendered PNG -> predicted exported BMP tensors -> expected BMP pixels

The model does not predict or train against padded atlas space. Atlas assembly is
only an export convenience after the trainable BMP tensors have been predicted.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from atlas_ai.export_spec import TRAINABLE_EXPORT_SPECS, ExportFileSpec


def _groups(channels: int, preferred: int) -> int:
    return preferred if channels % preferred == 0 else 1


def conv_block(in_ch: int, out_ch: int, preferred_groups: int, stride: int = 1) -> nn.Sequential:
    groups = _groups(out_ch, preferred_groups)
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
        nn.GroupNorm(groups, out_ch),
        nn.SiLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.GroupNorm(groups, out_ch),
        nn.SiLU(inplace=True),
    )


def _coord_channels(frequencies: tuple[int, ...]) -> int:
    return 2 + 4 * len(frequencies)


def _head_key(file_name: str) -> str:
    return file_name.lower().removesuffix(".bmp")


def build_fourier_coords(height: int, width: int, frequencies: tuple[int, ...]) -> torch.Tensor:
    y01 = torch.linspace(0.0, 1.0, height)
    x01 = torch.linspace(0.0, 1.0, width)
    yy01, xx01 = torch.meshgrid(y01, x01, indexing="ij")
    channels = [xx01 * 2.0 - 1.0, yy01 * 2.0 - 1.0]
    for freq in frequencies:
        angle_x = 2.0 * math.pi * freq * xx01
        angle_y = 2.0 * math.pi * freq * yy01
        channels.extend((torch.sin(angle_x), torch.cos(angle_x), torch.sin(angle_y), torch.cos(angle_y)))
    return torch.stack(channels, dim=0)


class ExportFileHead(nn.Module):
    def __init__(
        self,
        spec: ExportFileSpec,
        style_dim: int,
        hidden_channels: int,
        frequencies: tuple[int, ...],
    ):
        super().__init__()
        self.spec = spec
        self.style_to_map = nn.Linear(style_dim, hidden_channels)
        in_channels = hidden_channels + _coord_channels(frequencies)
        groups = 8 if hidden_channels % 8 == 0 else 1
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1),
            nn.GroupNorm(groups, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(groups, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(groups, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, 3, 1),
        )
        self.register_buffer("coords", build_fourier_coords(spec.h, spec.w, frequencies))

    def forward(self, style: torch.Tensor) -> torch.Tensor:
        batch = style.shape[0]
        style_map = self.style_to_map(style).to(dtype=self.coords.dtype)
        style_map = style_map[:, :, None, None].expand(-1, -1, self.spec.h, self.spec.w)
        coords = self.coords.unsqueeze(0).expand(batch, -1, -1, -1)
        x = torch.cat((style_map, coords.to(dtype=style_map.dtype)), dim=1)
        return self.body(x)


class SlotNetV35(nn.Module):
    def __init__(
        self,
        base_channels: int = 24,
        style_dim: int = 192,
        head_channels: int | None = None,
        frequencies: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64),
    ):
        super().__init__()
        head_channels = head_channels or base_channels * 3

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 6
        c5 = base_channels * 8

        self.enc1 = conv_block(3, c1, 8, stride=1)
        self.enc2 = conv_block(c1, c2, 8, stride=2)
        self.enc3 = conv_block(c2, c3, 16, stride=2)
        self.enc4 = conv_block(c3, c4, 16, stride=2)
        self.enc5 = conv_block(c4, c5, 32, stride=2)
        self.style_proj = nn.Sequential(
            nn.Linear(c5, style_dim),
            nn.SiLU(inplace=True),
            nn.Linear(style_dim, style_dim),
        )
        self.heads = nn.ModuleDict(
            {
                _head_key(spec.file_name): ExportFileHead(
                    spec=spec,
                    style_dim=style_dim,
                    hidden_channels=head_channels,
                    frequencies=frequencies,
                )
                for spec in TRAINABLE_EXPORT_SPECS
            }
        )
        self.register_buffer("slotnet_version", torch.tensor([35], dtype=torch.int32))

    def encode(self, view: torch.Tensor) -> torch.Tensor:
        f1 = self.enc1(view)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        f4 = self.enc4(f3)
        f5 = self.enc5(f4)
        return self.style_proj(f5.mean(dim=(2, 3)))

    def forward(self, view: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
        style = self.encode(view)
        return {"files": {head.spec.file_name: head(style) for head in self.heads.values()}}


__all__ = ["SlotNetV35", "ExportFileHead", "build_fourier_coords"]
