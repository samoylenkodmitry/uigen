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
    style z    [B, D] optional deployable code derived from source_rgb
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


# Output-head parameterizations (compared in the S1 head A/B):
#   residual  - clamp(source + tanh(delta)); bounded residual (the original).
#   direct    - sigmoid(logits); predict the target outright, no residual base.
#   unbounded - clamp(source + delta); residual with no tanh bound (diagnostic).
#   gated     - out = (1-g)*source + g*sigmoid(rgb); g a learned per-pixel
#               overwrite gate. Gate bias inits negative so it starts near copy
#               (slider strength) and opens only where the state changes
#               (high-contrast precision). Combines residual + direct.
_OUTPUT_MODES = {"residual": 0, "direct": 1, "unbounded": 2, "gated": 3}
_GATE_BIAS_INIT = -2.0
_OUTPUT_MODE_BY_CODE = {v: k for k, v in _OUTPUT_MODES.items()}


class _SourceStyleEncoder(nn.Module):
    """Small source-frame encoder for deployable style/context conditioning."""

    def __init__(self, out_dim: int, base_channels: int):
        super().__init__()
        width = max(8, min(int(base_channels), int(out_dim)))
        self.features = nn.Sequential(
            _conv_block(3, width),
            _conv_block(width, width, stride=2),
            _conv_block(width, width, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        # RGB mean/std gives the code a direct low-frequency style path while
        # the conv tower captures local material/edge texture.
        self.proj = nn.Sequential(
            nn.Linear(width + 6, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(inplace=True),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, source_rgb: torch.Tensor) -> torch.Tensor:
        feat = self.pool(self.features(source_rgb)).flatten(1)
        rgb = source_rgb.flatten(2)
        stats = torch.cat(
            [rgb.mean(dim=2), rgb.std(dim=2, unbiased=False)],
            dim=1,
        )
        return self.proj(torch.cat([feat, stats], dim=1))


class _GeometryGate(nn.Module):
    """Per-pixel, RGB-FREE gate prior — a coordinate MLP (1x1 convs) over
    [fourier coords, broadcast pair-geometry scalars, broadcast family code].

    Every input is skin-INDEPENDENT: classic sprite geometry is fixed across
    skins and family ids are shared, so the change-localization it learns
    transfers to unseen skins — unlike the content gate, which is produced from
    RGB-entangled U-Net features and stays out-of-distribution on a new style
    (the S2a / S2a-context failure: the gate never opened on held-out skins).

    Its logits are ADDED to the content gate's logits. The final layer is
    zero-initialized so the prior starts as a no-op (total gate == content gate)
    and learns a positive additive correction in the regions a transition
    changes."""

    def __init__(self, in_ch: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, geo_channels: torch.Tensor) -> torch.Tensor:
        return self.net(geo_channels)


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
        style_context_dim: deployable source-derived style/context width. When
            >0, the model encodes source_rgb to a global code and broadcasts it
            into the transition/gate path. This is not an oracle id and is safe
            for unseen skins.
        geometry_gate: when True (gated mode only) add a skin-independent
            additive gate prior derived from fixed classic geometry + family id
            (no RGB). Targets the S2a failure where the content gate, built from
            RGB features, never opened on unseen skins. Expects forward(...,
            pair_geom=...).
        geo_gate_hidden: hidden width of the geometry-gate coordinate MLP.
        geometry_dim: width of the pair_geom vector (must match the dataset's
            PAIR_GEOM_DIM).
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
        style_context_dim: int = 0,
        geometry_gate: bool = False,
        geo_gate_hidden: int = 64,
        geometry_dim: int = 13,
        frequencies: tuple[int, ...] = DEFAULT_FREQUENCIES,
        output_mode: str = "gated",
    ):
        super().__init__()
        if output_mode not in _OUTPUT_MODES:
            raise ValueError(f"output_mode must be one of {list(_OUTPUT_MODES)}, got {output_mode!r}")
        if geometry_gate and output_mode != "gated":
            raise ValueError("geometry_gate requires output_mode='gated' (it adds to the gate logits)")
        self.output_mode = output_mode
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
        self.style_context_dim = int(style_context_dim)
        if self.style_context_dim < 0:
            raise ValueError("style_context_dim must be >=0")
        self.geometry_gate = bool(geometry_gate)
        self.geo_gate_hidden = int(geo_gate_hidden)
        self.geometry_dim = int(geometry_dim)
        self.frequencies = tuple(int(f) for f in frequencies)

        self.file_embedding = nn.Embedding(len(TRAINABLE_EXPORT_SPECS), self.file_embedding_dim)
        self.family_embedding = nn.Embedding(self.num_families, self.family_embedding_dim)
        # Shared frame table for source/target idx so the two share a vocabulary.
        self.frame_embedding = nn.Embedding(self.max_frames, self.frame_embedding_dim)
        if self.num_skins > 0:
            self.skin_embedding = nn.Embedding(self.num_skins, self.skin_embedding_dim)
        else:
            self.skin_embedding = None
        if self.style_context_dim > 0:
            self.style_context_encoder = _SourceStyleEncoder(
                self.style_context_dim, base_channels=self.base_channels
            )
        else:
            self.style_context_encoder = None

        coord_ch = _coord_channels(self.frequencies)
        in_ch = (
            3
            + coord_ch
            + self.file_embedding_dim
            + self.family_embedding_dim
            + 2 * self.frame_embedding_dim
            + self.skin_embedding_dim
            + self.style_context_dim
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
        # Gated mode adds a per-pixel overwrite gate head, biased toward copy.
        # Only built for output_mode=="gated" so other modes' state_dicts are
        # unchanged (older checkpoints still load).
        if output_mode == "gated":
            self.gate_proj = nn.Conv2d(c1, 1, kernel_size=1)
            nn.init.constant_(self.gate_proj.bias, _GATE_BIAS_INIT)
        else:
            self.gate_proj = None
        # Geometry gate prior: a skin-independent additive logit on the gate.
        if self.geometry_gate:
            geo_in = coord_ch + self.geometry_dim + self.family_embedding_dim
            self.geo_gate = _GeometryGate(geo_in, self.geo_gate_hidden)
        else:
            self.geo_gate = None

        model_version = 74 if self.geometry_gate else (73 if self.style_context_dim > 0 else 72)
        self.register_buffer("model_version", torch.tensor([model_version], dtype=torch.int32))
        self.register_buffer(
            "output_mode_buffer", torch.tensor([_OUTPUT_MODES[output_mode]], dtype=torch.int32)
        )
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
        if self.style_context_dim > 0:
            self.register_buffer(
                "style_context_dim_buffer", torch.tensor([self.style_context_dim], dtype=torch.int32)
            )
        # Geometry-gate buffers registered only when enabled, so older v72/v73
        # checkpoints (no geometry gate) still load into a default-built model.
        if self.geometry_gate:
            for name, val in (
                ("geometry_gate_buffer", 1),
                ("geo_gate_hidden_buffer", self.geo_gate_hidden),
                ("geometry_dim_buffer", self.geometry_dim),
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
        if self.style_context_encoder is not None:
            channels.append(self._map(self.style_context_encoder(source_rgb), b, h, w))
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
        pair_geom: torch.Tensor | None = None,
        return_gate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if source_rgb.dim() != 4 or source_rgb.shape[1] != 3:
            raise ValueError(f"source_rgb must be [B, 3, H, W], got {tuple(source_rgb.shape)}")
        b, _, h, w = source_rgb.shape
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
        out = self.out_proj(u1)
        gate = gate_logits = None
        if self.output_mode == "direct":
            # Predict the target outright; source is still an input channel so
            # the net can copy, but it must learn to.
            result = torch.sigmoid(out)
        elif self.output_mode == "gated":
            rgb = torch.sigmoid(out)
            gate_logits = self.gate_proj(u1)            # [B, 1, H, W] content (RGB) gate
            if self.geo_gate is not None:
                # Skin-independent additive gate prior: localize the change from
                # fixed classic geometry, not from the (OOD-on-unseen-skins) RGB
                # features. coords + broadcast pair-geometry + broadcast family.
                if pair_geom is None:
                    raise ValueError("pair_geom is required when geometry_gate=True")
                coords = _fourier_coords(h, w, self.frequencies, source_rgb.device, source_rgb.dtype)
                geo_channels = torch.cat([
                    coords.unsqueeze(0).expand(b, -1, -1, -1),
                    self._map(pair_geom, b, h, w),
                    self._map(self.family_embedding(family_id), b, h, w),
                ], dim=1)
                gate_logits = gate_logits + self.geo_gate(geo_channels)
            gate = torch.sigmoid(gate_logits)            # copy-biased gate in [0, 1]
            result = (1.0 - gate) * source_rgb + gate * rgb
        else:
            # Residual from source: unchanged content is an exact copy when delta=0.
            delta = torch.tanh(out) if self.output_mode == "residual" else out
            result = (source_rgb + delta).clamp(0.0, 1.0)
        # return_gate -> (result, gate[0,1], gate_logits); gate/logits None unless gated.
        return (result, gate, gate_logits) if return_gate else result


__all__ = ["V7StateExpander"]
