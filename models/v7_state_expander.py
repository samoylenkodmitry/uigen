"""V7.1 StateFamilyExpander.

Split-by-task replacement for the generic completer's `state_family` mode. The
generic U-Net could not reproduce crisp hidden pixels (see the MAIN floor
test). State expansion is a far more constrained problem: an `alternatives`
family is a low-dimensional manifold (one parameter — which frame), so a model
conditioned on (source frame, source_idx, target_idx, family_id, file_id) can
learn the transition directly.

    source_rgb [B, 3, H, W]  (the frame we are shown)
    source_idx [B]           (which frame it is)
    target_idx [B]           (which frame we want)
    family_id  [B]           (which alternatives family)
    file_id    [B]           (which BMP)
    skin_id    [B] optional   ORACLE skin id (capacity test only, not deployable)
  ->
    target_rgb [B, 3, H, W]  (the requested frame)

Output is residual from the source: target = clamp(source + delta). State
frames are mostly identical except where the state changes (slider thumb moves,
button darkens, indicator lights), so a residual base gives the model an easy
exact copy of the unchanged majority — exactly the crisp-copy property the
generic completer lacked — and concentrates capacity on the local change.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from atlas_ai.export_spec import TRAINABLE_EXPORT_SPECS
from models.v7_completer import (
    DEFAULT_FREQUENCIES,
    _conv_block,
    _coord_channels,
    _fourier_coords,
)


class V7StateExpander(nn.Module):
    """Conditional frame/state translator for alternatives families.

    Args:
        num_families: number of alternatives families (global family_id range).
        max_frames: max frame count across families (frame embedding size).
        base_channels: first-encoder width; doubles to *8 at the bottleneck.
        file_embedding_dim / family_embedding_dim / frame_embedding_dim:
            conditioning embedding widths (broadcast as feature maps).
        num_skins: ORACLE skin embedding table size; 0 disables skin
            conditioning. >0 expects forward(..., skin_id=...).
        skin_embedding_dim: skin embedding width when num_skins>0.
        frequencies: Fourier frequencies for spatial coords.
    """

    def __init__(
        self,
        *,
        num_families: int,
        max_frames: int,
        base_channels: int = 48,
        file_embedding_dim: int = 16,
        family_embedding_dim: int = 16,
        frame_embedding_dim: int = 16,
        num_skins: int = 0,
        skin_embedding_dim: int = 0,
        frequencies: tuple[int, ...] = DEFAULT_FREQUENCIES,
    ):
        super().__init__()
        self.num_families = int(num_families)
        self.max_frames = int(max_frames)
        self.base_channels = int(base_channels)
        self.file_embedding_dim = int(file_embedding_dim)
        self.family_embedding_dim = int(family_embedding_dim)
        self.frame_embedding_dim = int(frame_embedding_dim)
        self.num_skins = int(num_skins)
        self.skin_embedding_dim = int(skin_embedding_dim) if self.num_skins > 0 else 0
        if self.num_skins > 0 and self.skin_embedding_dim <= 0:
            raise ValueError("skin_embedding_dim must be >0 when num_skins>0")
        self.frequencies = tuple(int(f) for f in frequencies)

        self.file_embedding = nn.Embedding(len(TRAINABLE_EXPORT_SPECS), self.file_embedding_dim)
        self.family_embedding = nn.Embedding(self.num_families, self.family_embedding_dim)
        # Shared frame table for source/target idx so the two share a vocabulary.
        self.frame_embedding = nn.Embedding(self.max_frames, self.frame_embedding_dim)
        if self.num_skins > 0:
            self.skin_embedding = nn.Embedding(self.num_skins, self.skin_embedding_dim)
        else:
            self.skin_embedding = None

        coord_ch = _coord_channels(self.frequencies)
        in_ch = (
            3
            + coord_ch
            + self.file_embedding_dim
            + self.family_embedding_dim
            + 2 * self.frame_embedding_dim
            + self.skin_embedding_dim
        )

        c1, c2, c3, c4 = (base_channels, base_channels * 2, base_channels * 4, base_channels * 8)
        self.stem = _conv_block(in_ch, c1)
        self.down1 = _conv_block(c1, c2, stride=2)
        self.down2 = _conv_block(c2, c3, stride=2)
        self.down3 = _conv_block(c3, c4, stride=2)
        self.fuse3 = _conv_block(c4 + c3, c3)
        self.fuse2 = _conv_block(c3 + c2, c2)
        self.fuse1 = _conv_block(c2 + c1, c1)
        self.out_proj = nn.Conv2d(c1, 3, kernel_size=1)

        self.register_buffer("model_version", torch.tensor([72], dtype=torch.int32))
        for name, val in (
            ("num_families_buffer", self.num_families),
            ("max_frames_buffer", self.max_frames),
            ("base_channels_buffer", self.base_channels),
            ("file_embedding_dim_buffer", self.file_embedding_dim),
            ("family_embedding_dim_buffer", self.family_embedding_dim),
            ("frame_embedding_dim_buffer", self.frame_embedding_dim),
            ("num_skins_buffer", self.num_skins),
            ("skin_embedding_dim_buffer", self.skin_embedding_dim),
        ):
            self.register_buffer(name, torch.tensor([val], dtype=torch.int32))
        self.register_buffer(
            "frequencies_buffer", torch.tensor(list(self.frequencies), dtype=torch.int32)
        )

    def _map(self, emb: torch.Tensor, b: int, h: int, w: int) -> torch.Tensor:
        return emb.view(b, -1, 1, 1).expand(b, -1, h, w)

    def _build_conditioning(
        self,
        source_rgb: torch.Tensor,
        source_idx: torch.Tensor,
        target_idx: torch.Tensor,
        family_id: torch.Tensor,
        file_id: torch.Tensor,
        skin_id: torch.Tensor | None,
    ) -> torch.Tensor:
        b, _, h, w = source_rgb.shape
        channels = [
            source_rgb,
            self._map(self.file_embedding(file_id), b, h, w),
            self._map(self.family_embedding(family_id), b, h, w),
            self._map(self.frame_embedding(source_idx), b, h, w),
            self._map(self.frame_embedding(target_idx), b, h, w),
        ]
        if self.skin_embedding is not None:
            if skin_id is None:
                raise ValueError("skin_id is required when num_skins>0")
            channels.append(self._map(self.skin_embedding(skin_id), b, h, w))
        coords = _fourier_coords(h, w, self.frequencies, source_rgb.device, source_rgb.dtype)
        channels.append(coords.unsqueeze(0).expand(b, -1, -1, -1))
        return torch.cat(channels, dim=1)

    def forward(
        self,
        source_rgb: torch.Tensor,
        source_idx: torch.Tensor,
        target_idx: torch.Tensor,
        family_id: torch.Tensor,
        file_id: torch.Tensor,
        skin_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if source_rgb.dim() != 4 or source_rgb.shape[1] != 3:
            raise ValueError(f"source_rgb must be [B, 3, H, W], got {tuple(source_rgb.shape)}")
        x = self._build_conditioning(
            source_rgb, source_idx, target_idx, family_id, file_id, skin_id
        )
        f1 = self.stem(x)
        f2 = self.down1(f1)
        f3 = self.down2(f2)
        f4 = self.down3(f3)
        u3 = self.fuse3(torch.cat([F.interpolate(f4, size=f3.shape[-2:], mode="nearest"), f3], dim=1))
        u2 = self.fuse2(torch.cat([F.interpolate(u3, size=f2.shape[-2:], mode="nearest"), f2], dim=1))
        u1 = self.fuse1(torch.cat([F.interpolate(u2, size=f1.shape[-2:], mode="nearest"), f1], dim=1))
        # Residual from source: unchanged content is an exact copy when delta=0.
        delta = torch.tanh(self.out_proj(u1))
        return (source_rgb + delta).clamp(0.0, 1.0)


__all__ = ["V7StateExpander"]
