"""V7 asset-completion model (Phase 0).

Phase 0 trains the V7 completer branch standalone on partial-evidence inputs,
without the observer/copy path. The model is a small masked U-Net that takes
per-file partial RGB + observation mask + file identity + Fourier coords and
outputs a complete clean BMP prediction at the file's exact dimensions.

Inputs per file:
    observed_rgb  [B, 3, H, W]   in [0, 1]; hidden pixels are zero
    observed_mask [B, 1, H, W]   in {0, 1}; 1 where the model is told the value
    file_id       [B]            int index into TRAINABLE_EXPORT_SPECS

Output:
    final_rgb     [B, 3, H, W]   in [0, 1]

Variable input shapes are handled by computing Fourier coords on the fly and
using `F.interpolate(..., size=skip.shape[-2:])` in the decoder so the skip
connections line up regardless of file dimensions.

Per the V7 plan, this is *not* a 1x1 head: it has a real encoder-decoder with
skip connections. Depth is intentionally shallow (3 downsample levels) so the
smallest trainable BMP (PLAYPAUS at 42x9) still has a meaningful bottleneck.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from atlas_ai.export_spec import TRAINABLE_EXPORT_SPECS


DEFAULT_FREQUENCIES = (1, 2, 4, 8)


def _groups(channels: int, preferred: int = 8) -> int:
    return preferred if channels % preferred == 0 else 1


def _conv_block(in_ch: int, out_ch: int, *, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
        nn.GroupNorm(_groups(out_ch), out_ch),
        nn.SiLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.GroupNorm(_groups(out_ch), out_ch),
        nn.SiLU(inplace=True),
    )


def _coord_channels(frequencies: tuple[int, ...]) -> int:
    return 2 + 4 * len(frequencies)


def _fourier_coords(
    h: int, w: int, frequencies: tuple[int, ...],
    device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    """Compute [coord_channels, H, W] Fourier coords for one image size."""
    y01 = torch.linspace(0.0, 1.0, h, device=device, dtype=dtype)
    x01 = torch.linspace(0.0, 1.0, w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y01, x01, indexing="ij")
    channels = [xx * 2.0 - 1.0, yy * 2.0 - 1.0]
    for freq in frequencies:
        angle_x = 2.0 * math.pi * freq * xx
        angle_y = 2.0 * math.pi * freq * yy
        channels.extend((
            torch.sin(angle_x), torch.cos(angle_x),
            torch.sin(angle_y), torch.cos(angle_y),
        ))
    return torch.stack(channels, dim=0)


class V7Completer(nn.Module):
    """Small masked U-Net for asset completion.

    Args:
        base_channels: width of the first encoder block. Each level doubles up
            to base_channels * 8 at the bottleneck.
        file_embedding_dim: embedding width for the per-file conditioning
            channel. Broadcast over all spatial positions.
        frequencies: Fourier frequencies for spatial coord conditioning.
    """

    def __init__(
        self,
        base_channels: int = 24,
        file_embedding_dim: int = 32,
        frequencies: tuple[int, ...] = DEFAULT_FREQUENCIES,
    ):
        super().__init__()
        self.base_channels = base_channels
        self.file_embedding_dim = file_embedding_dim
        self.frequencies = tuple(int(f) for f in frequencies)
        coord_ch = _coord_channels(self.frequencies)
        in_ch = 3 + 1 + file_embedding_dim + coord_ch

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.file_embedding = nn.Embedding(
            len(TRAINABLE_EXPORT_SPECS), file_embedding_dim
        )

        # Encoder: stem at full resolution, then 3 stride-2 down blocks.
        self.stem = _conv_block(in_ch, c1)
        self.down1 = _conv_block(c1, c2, stride=2)
        self.down2 = _conv_block(c2, c3, stride=2)
        self.down3 = _conv_block(c3, c4, stride=2)

        # Decoder: each level fuses upsampled features with the corresponding
        # encoder skip via F.interpolate to the skip's spatial shape.
        self.fuse3 = _conv_block(c4 + c3, c3)
        self.fuse2 = _conv_block(c3 + c2, c2)
        self.fuse1 = _conv_block(c2 + c1, c1)
        self.out_proj = nn.Conv2d(c1, 3, kernel_size=1)

        self.register_buffer("model_version", torch.tensor([70], dtype=torch.int32))
        self.register_buffer("base_channels_buffer", torch.tensor([base_channels], dtype=torch.int32))
        self.register_buffer("file_embedding_dim_buffer", torch.tensor([file_embedding_dim], dtype=torch.int32))
        self.register_buffer(
            "frequencies_buffer", torch.tensor(list(self.frequencies), dtype=torch.int32)
        )

    def _build_conditioning(
        self, observed_rgb: torch.Tensor, observed_mask: torch.Tensor, file_id: torch.Tensor,
    ) -> torch.Tensor:
        b, _, h, w = observed_rgb.shape
        emb = self.file_embedding(file_id)  # [B, F]
        emb_map = emb.view(b, -1, 1, 1).expand(b, -1, h, w)
        coords = _fourier_coords(
            h, w, self.frequencies,
            device=observed_rgb.device, dtype=observed_rgb.dtype,
        )
        coords = coords.unsqueeze(0).expand(b, -1, -1, -1)
        return torch.cat([observed_rgb, observed_mask, emb_map, coords], dim=1)

    def forward(
        self,
        observed_rgb: torch.Tensor,
        observed_mask: torch.Tensor,
        file_id: torch.Tensor,
    ) -> torch.Tensor:
        if observed_rgb.dim() != 4 or observed_rgb.shape[1] != 3:
            raise ValueError(f"observed_rgb must be [B, 3, H, W], got {tuple(observed_rgb.shape)}")
        if observed_mask.shape != (observed_rgb.shape[0], 1, *observed_rgb.shape[2:]):
            raise ValueError(
                f"observed_mask must be [B, 1, H, W], got {tuple(observed_mask.shape)} "
                f"vs view {tuple(observed_rgb.shape)}"
            )
        x = self._build_conditioning(observed_rgb, observed_mask, file_id)
        # Encoder.
        f1 = self.stem(x)
        f2 = self.down1(f1)
        f3 = self.down2(f2)
        f4 = self.down3(f3)
        # Decoder: upsample to each skip's spatial shape, fuse via concat.
        u3 = F.interpolate(f4, size=f3.shape[-2:], mode="nearest")
        u3 = self.fuse3(torch.cat([u3, f3], dim=1))
        u2 = F.interpolate(u3, size=f2.shape[-2:], mode="nearest")
        u2 = self.fuse2(torch.cat([u2, f2], dim=1))
        u1 = F.interpolate(u2, size=f1.shape[-2:], mode="nearest")
        u1 = self.fuse1(torch.cat([u1, f1], dim=1))
        return torch.sigmoid(self.out_proj(u1))


__all__ = ["V7Completer", "DEFAULT_FREQUENCIES"]
