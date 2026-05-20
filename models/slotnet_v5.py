"""SlotNetV5: per-file query grids cross-attending to encoder spatial tokens.

Same training contract as V3.5:

    input rendered PNG -> predicted exported BMP tensors -> expected BMP pixels

V5 keeps the V3.5 encoder, the global style vector, and the per-file decoder
output, but adds a local spatial evidence path: every per-file decoder owns a
low-resolution query grid that cross-attends into the encoder's spatial
feature map. The decoder receives encoder features sampled at the locations
the model thinks are relevant for that specific file, instead of relying on
the global mean-pooled style vector alone.

The output API matches V3.5:

    model(view) -> {"files": {file_name: logits}}

`forward(view, return_attention=True)` additionally returns per-file
attention summaries shaped (B, h_enc, w_enc), averaged across heads and
queries; intended for debug heatmaps only, never for loss.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from atlas_ai.export_spec import TRAINABLE_EXPORT_SPECS, ExportFileSpec

# Defaults chosen per HANDOFF V5; cross_attention_layers stays at 1.
DEFAULT_FREQUENCIES = (1, 2, 4, 8)
QUERY_GRID_DIVISOR = 8


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


def _query_grid_size(spec: ExportFileSpec) -> tuple[int, int]:
    return (
        max(1, math.ceil(spec.h / QUERY_GRID_DIVISOR)),
        max(1, math.ceil(spec.w / QUERY_GRID_DIVISOR)),
    )


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


class CrossAttentionHead(nn.Module):
    """Per-file query grid + cross-attention + nearest-upsample decoder.

    The file embedding lives on the parent module (`SlotNetV5.file_embedding`)
    and is supplied at forward time so safetensors can serialise a single copy.
    """

    def __init__(
        self,
        spec: ExportFileSpec,
        file_index: int,
        attn_dim: int,
        attention_heads: int,
        cross_attention_layers: int,
        style_dim: int,
        head_channels: int,
        file_embedding_dim: int,
        frequencies: tuple[int, ...] = DEFAULT_FREQUENCIES,
    ):
        super().__init__()
        self.spec = spec
        self.file_index = file_index
        self.attn_dim = attn_dim

        h0, w0 = _query_grid_size(spec)
        self.h0 = h0
        self.w0 = w0

        coord_channels = _coord_channels(frequencies)
        # Tokens-in: fourier coords + file embedding + style projection.
        token_in_channels = coord_channels + file_embedding_dim + style_dim
        self.query_proj = nn.Linear(token_in_channels, attn_dim)

        self.cross_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=attn_dim,
                    num_heads=attention_heads,
                    batch_first=True,
                )
                for _ in range(cross_attention_layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(attn_dim) for _ in range(cross_attention_layers)])
        self.ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(attn_dim, attn_dim * 2),
                    nn.SiLU(inplace=True),
                    nn.Linear(attn_dim * 2, attn_dim),
                )
                for _ in range(cross_attention_layers)
            ]
        )
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(attn_dim) for _ in range(cross_attention_layers)])

        groups = 8 if head_channels % 8 == 0 else 1
        self.decoder = nn.Sequential(
            nn.Conv2d(attn_dim, head_channels, 1),
            nn.GroupNorm(groups, head_channels),
            nn.SiLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(head_channels, head_channels, 3, padding=1),
            nn.GroupNorm(groups, head_channels),
            nn.SiLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(head_channels, head_channels, 3, padding=1),
            nn.GroupNorm(groups, head_channels),
            nn.SiLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(head_channels, head_channels, 3, padding=1),
            nn.GroupNorm(groups, head_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(head_channels, 3, 1),
        )

        # Static Fourier coords for the query grid; small, register as a buffer.
        self.register_buffer(
            "grid_coords",
            build_fourier_coords(h0, w0, frequencies).reshape(coord_channels, -1).T.contiguous(),
        )

    def build_queries(self, style: torch.Tensor, file_emb: torch.Tensor) -> torch.Tensor:
        batch = style.shape[0]
        device = style.device
        # Fourier coords: [h0*w0, coord_channels] -> [B, h0*w0, coord_channels]
        coords = self.grid_coords.to(dtype=style.dtype, device=device)
        coords = coords.unsqueeze(0).expand(batch, -1, -1)
        # file_emb: [file_emb_dim] -> [B, h0*w0, file_emb_dim]
        emb = file_emb.to(dtype=style.dtype).view(1, 1, -1).expand(batch, self.h0 * self.w0, -1)
        # Global style broadcast over the grid.
        style_per_cell = style.unsqueeze(1).expand(-1, self.h0 * self.w0, -1)
        combined = torch.cat((coords, emb, style_per_cell), dim=-1)
        return self.query_proj(combined)

    def forward(
        self,
        style: torch.Tensor,
        encoder_tokens: torch.Tensor,
        file_emb: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        queries = self.build_queries(style, file_emb)
        attn_summary: torch.Tensor | None = None
        for layer_idx, (attn, norm, ffn, ffn_norm) in enumerate(
            zip(self.cross_layers, self.norms, self.ffns, self.ffn_norms)
        ):
            normed = norm(queries)
            want_weights = return_attention and layer_idx == len(self.cross_layers) - 1
            attended, weights = attn(
                normed,
                encoder_tokens,
                encoder_tokens,
                need_weights=want_weights,
                average_attn_weights=True,
            )
            queries = queries + attended
            queries = queries + ffn(ffn_norm(queries))
            if want_weights and weights is not None:
                # weights: [B, num_queries, num_keys]; mean over queries -> per-key heatmap.
                attn_summary = weights.mean(dim=1)
        batch = queries.shape[0]
        spatial = queries.transpose(1, 2).reshape(batch, self.attn_dim, self.h0, self.w0)
        logits = self.decoder(spatial)
        if logits.shape[-2:] != (self.spec.h, self.spec.w):
            logits = F.interpolate(logits, size=(self.spec.h, self.spec.w), mode="nearest")
        return logits, attn_summary


class SlotNetV5(nn.Module):
    def __init__(
        self,
        base_channels: int = 24,
        style_dim: int = 192,
        head_channels: int | None = None,
        attn_dim: int = 128,
        attention_heads: int = 4,
        cross_attention_layers: int = 1,
        file_embedding_dim: int = 32,
        frequencies: tuple[int, ...] = DEFAULT_FREQUENCIES,
    ):
        super().__init__()
        if attn_dim % attention_heads != 0:
            raise ValueError(
                f"attn_dim={attn_dim} must be divisible by attention_heads={attention_heads}"
            )
        head_channels = head_channels or base_channels * 4

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

        # Encoder feature map -> attention tokens.
        self.feature_proj = nn.Conv2d(c5, attn_dim, kernel_size=1)
        self.attn_dim = attn_dim
        # 2D positional encoding is recomputed lazily and cached on the module.
        # The table is fully determined by (h, w, attn_dim) so it does not need
        # to be a parameter or a buffer.
        self._pe_cache: dict[tuple[int, int, torch.device, torch.dtype], torch.Tensor] = {}

        self.file_embedding = nn.Embedding(len(TRAINABLE_EXPORT_SPECS), file_embedding_dim)

        self.heads = nn.ModuleDict(
            {
                _head_key(spec.file_name): CrossAttentionHead(
                    spec=spec,
                    file_index=idx,
                    attn_dim=attn_dim,
                    attention_heads=attention_heads,
                    cross_attention_layers=cross_attention_layers,
                    style_dim=style_dim,
                    head_channels=head_channels,
                    file_embedding_dim=file_embedding_dim,
                    frequencies=frequencies,
                )
                for idx, spec in enumerate(TRAINABLE_EXPORT_SPECS)
            }
        )

        self.register_buffer("slotnet_version", torch.tensor([50], dtype=torch.int32))
        self.register_buffer("attn_dim_buffer", torch.tensor([attn_dim], dtype=torch.int32))
        self.register_buffer("attention_heads_buffer", torch.tensor([attention_heads], dtype=torch.int32))
        self.register_buffer(
            "cross_attention_layers_buffer",
            torch.tensor([cross_attention_layers], dtype=torch.int32),
        )
        self.register_buffer(
            "frequencies_buffer",
            torch.tensor(list(frequencies), dtype=torch.int32),
        )

    def _ensure_pos_table(self, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (h, w, device, dtype)
        cached = self._pe_cache.get(key)
        if cached is not None:
            return cached
        pe = _build_sinusoidal_2d_pe(h, w, self.attn_dim).to(device=device, dtype=dtype)
        self._pe_cache[key] = pe
        return pe

    def encode_features(self, view: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f1 = self.enc1(view)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        f4 = self.enc4(f3)
        f5 = self.enc5(f4)
        return f5, self.style_proj(f5.mean(dim=(2, 3)))

    def feature_tokens(self, f5: torch.Tensor) -> torch.Tensor:
        projected = self.feature_proj(f5)
        batch, _, h_enc, w_enc = projected.shape
        pos = self._ensure_pos_table(h_enc, w_enc, projected.device, projected.dtype)
        tokens = projected.flatten(2).transpose(1, 2)
        return tokens + pos.unsqueeze(0)

    def forward(
        self,
        view: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> dict[str, dict[str, torch.Tensor]]:
        f5, style = self.encode_features(view)
        tokens = self.feature_tokens(f5)
        # Look up every file embedding once; pass the per-file vector to its head.
        all_file_emb = self.file_embedding(
            torch.arange(len(self.heads), device=view.device)
        )
        outputs: dict[str, torch.Tensor] = {}
        attention: dict[str, torch.Tensor] = {}
        h_enc, w_enc = f5.shape[-2:]
        for head in self.heads.values():
            logits, attn_summary = head(
                style,
                tokens,
                all_file_emb[head.file_index],
                return_attention=return_attention,
            )
            outputs[head.spec.file_name] = logits
            if return_attention and attn_summary is not None:
                attention[head.spec.file_name] = attn_summary.reshape(-1, h_enc, w_enc)
        if return_attention:
            return {"files": outputs, "attention": attention}
        return {"files": outputs}


def _build_sinusoidal_2d_pe(h: int, w: int, dim: int) -> torch.Tensor:
    if dim % 4 != 0:
        raise ValueError(f"positional encoding dim must be divisible by 4, got {dim}")
    half = dim // 2
    pe = torch.zeros(h, w, dim)
    div_term = torch.exp(torch.arange(0, half, 2, dtype=torch.float32) * -(math.log(10000.0) / half))
    y_pos = torch.arange(h, dtype=torch.float32).unsqueeze(1)
    x_pos = torch.arange(w, dtype=torch.float32).unsqueeze(1)
    pe_y = torch.zeros(h, half)
    pe_y[:, 0::2] = torch.sin(y_pos * div_term)
    pe_y[:, 1::2] = torch.cos(y_pos * div_term)
    pe_x = torch.zeros(w, half)
    pe_x[:, 0::2] = torch.sin(x_pos * div_term)
    pe_x[:, 1::2] = torch.cos(x_pos * div_term)
    pe[:, :, :half] = pe_y.unsqueeze(1).expand(h, w, half)
    pe[:, :, half:] = pe_x.unsqueeze(0).expand(h, w, half)
    return pe.reshape(h * w, dim)


__all__ = ["SlotNetV5", "CrossAttentionHead", "build_fourier_coords"]
