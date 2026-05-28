"""V10 BMPExpertNet: full render -> exact BMP tensor.

Per the HANDOFF_V10 architecture:

    Encoder:  4-stage CNN (strides 2/4/8/16) + 1x1 FPN to attn_dim
    Decoder:  target H/4 x W/4 query grid with Fourier x/y coords ->
              2x cross-attention into all encoder feature levels -> nearest
              upsample to target H x W + residual conv blocks -> RGB head.

    No bilinear at the final decoder (nearest + residual conv).

One model per output BMP; target_h/target_w come from atlas_ai.export_spec.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn_groups(c: int) -> int:
    """Largest divisor of c that is <= 8 (GroupNorm needs num_groups | c)."""
    for g in (8, 4, 2, 1):
        if c % g == 0:
            return g
    return 1


def _conv_block(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
        nn.GroupNorm(num_groups=_gn_groups(out_ch), num_channels=out_ch),
        nn.SiLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.GroupNorm(num_groups=_gn_groups(out_ch), num_channels=out_ch),
        nn.SiLU(inplace=True),
    )


class _ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.body = _conv_block(in_ch, out_ch)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.body(x) + self.skip(x)


def _fourier_coords(h: int, w: int, freqs: tuple[int, ...],
                    device, dtype) -> torch.Tensor:
    """[2 + 4*len(freqs), h, w]: raw x/y in [-1,1] + sin/cos at each freq."""
    ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype).view(h, 1).expand(h, w)
    xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype).view(1, w).expand(h, w)
    chans = [xs, ys]
    for f in freqs:
        chans += [torch.sin(math.pi * f * xs), torch.cos(math.pi * f * xs),
                  torch.sin(math.pi * f * ys), torch.cos(math.pi * f * ys)]
    return torch.stack(chans, dim=0)


class _CrossAttnBlock(nn.Module):
    """Pre-norm cross-attention + FFN (queries attend to encoder K/V)."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.qn = nn.LayerNorm(dim)
        self.kn = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.fn = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.SiLU(inplace=True), nn.Linear(dim * 2, dim))

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        qn = self.qn(q)
        kn = self.kn(kv)
        out, _ = self.attn(qn, kn, kn, need_weights=False)
        q = q + out
        q = q + self.ff(self.fn(q))
        return q


class BMPExpertNet(nn.Module):
    """Full render -> one exact BMP tensor.

    Args:
        target_h, target_w: output BMP dims (from atlas_ai.export_spec).
        base, attn_dim, dec_ch, heads, attn_layers: handoff starting config.
        freqs: Fourier coordinate frequencies for query positions.
    """

    DEFAULT_FREQS = (1, 2, 4, 8, 16, 32)
    # Adaptive KV pool per FPN level (h, w). Cross-attention over the full
    # feature pyramid would be ~5e5 tokens at 1728x960 -> intractable. Pooling
    # each level to a fixed (h, w) keeps multi-scale structure with ~2.4k total
    # KV tokens, which both CPU smoke and GPU training can handle.
    DEFAULT_KV_POOL = ((48, 28), (32, 18), (24, 14), (18, 10))

    def __init__(self, target_h: int, target_w: int, *,
                 base: int = 48, attn_dim: int = 256, dec_ch: int = 128,
                 heads: int = 4, attn_layers: int = 2,
                 freqs: tuple[int, ...] = DEFAULT_FREQS,
                 kv_pool: tuple[tuple[int, int], ...] = DEFAULT_KV_POOL):
        super().__init__()
        self.target_h = int(target_h)
        self.target_w = int(target_w)
        self.attn_dim = attn_dim
        self.freqs = tuple(int(f) for f in freqs)
        self.kv_pool = tuple((int(h), int(w)) for (h, w) in kv_pool)
        if len(self.kv_pool) != 4:
            raise ValueError("kv_pool must have one (h, w) per encoder level (4 total)")

        # Encoder: 4 stages, strides 2/4/8/16 relative to input.
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8
        self.stem = _conv_block(3, c1, stride=2)         # /2
        self.e1 = _conv_block(c1, c2, stride=2)          # /4
        self.e2 = _conv_block(c2, c3, stride=2)          # /8
        self.e3 = _conv_block(c3, c4, stride=2)          # /16

        # FPN: project each level to attn_dim for cross-attention K/V.
        self.proj = nn.ModuleList([
            nn.Conv2d(c1, attn_dim, 1),
            nn.Conv2d(c2, attn_dim, 1),
            nn.Conv2d(c3, attn_dim, 1),
            nn.Conv2d(c4, attn_dim, 1),
        ])

        # Query embedding (Fourier coords -> attn_dim).
        coord_ch = 2 + 4 * len(self.freqs)
        self.query_proj = nn.Sequential(nn.Linear(coord_ch, attn_dim),
                                         nn.SiLU(inplace=True),
                                         nn.Linear(attn_dim, attn_dim))

        # Cross-attention stack.
        self.attn_blocks = nn.ModuleList([_CrossAttnBlock(attn_dim, heads)
                                          for _ in range(attn_layers)])

        # Decoder: nearest upsample from query grid to target HxW + residual conv blocks.
        self.dec_proj = nn.Conv2d(attn_dim, dec_ch, 1)
        self.up1 = _ResBlock(dec_ch, dec_ch)
        self.up2 = _ResBlock(dec_ch, dec_ch)
        self.head = nn.Conv2d(dec_ch, 3, 1)

        # Buffers for reconstructibility from a checkpoint alone.
        self.register_buffer("model_version", torch.tensor([10], dtype=torch.int32))
        self.register_buffer("target_h_buf", torch.tensor([self.target_h], dtype=torch.int32))
        self.register_buffer("target_w_buf", torch.tensor([self.target_w], dtype=torch.int32))
        for name, val in (("base_buf", base), ("attn_dim_buf", attn_dim),
                          ("dec_ch_buf", dec_ch), ("heads_buf", heads),
                          ("attn_layers_buf", attn_layers)):
            self.register_buffer(name, torch.tensor([val], dtype=torch.int32))

    def _query_grid(self, b: int, device, dtype) -> torch.Tensor:
        qh = max(1, self.target_h // 4)
        qw = max(1, self.target_w // 4)
        coords = _fourier_coords(qh, qw, self.freqs, device, dtype)  # [C, qh, qw]
        coords = coords.permute(1, 2, 0).reshape(qh * qw, -1)         # [qh*qw, C]
        q = self.query_proj(coords).unsqueeze(0).expand(b, -1, -1)    # [B, qh*qw, attn_dim]
        return q, qh, qw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.shape[1] != 3:
            raise ValueError(f"x must be [B,3,H,W], got {tuple(x.shape)}")
        b = x.shape[0]
        f1 = self.stem(x)
        f2 = self.e1(f1)
        f3 = self.e2(f2)
        f4 = self.e3(f3)
        feats = []
        for i, f in enumerate((f1, f2, f3, f4)):
            f = self.proj[i](f)
            f = F.adaptive_avg_pool2d(f, self.kv_pool[i])
            feats.append(f)
        # K/V = concat pooled spatial tokens across scales (~2.4k tokens default).
        kv = torch.cat([fi.flatten(2).transpose(1, 2) for fi in feats], dim=1)  # [B, sumHW, D]
        q, qh, qw = self._query_grid(b, x.device, x.dtype)
        for blk in self.attn_blocks:
            q = blk(q, kv)
        q = q.transpose(1, 2).reshape(b, self.attn_dim, qh, qw)         # [B, D, qh, qw]
        q = self.dec_proj(q)                                             # [B, dec_ch, qh, qw]
        q = F.interpolate(q, size=(self.target_h, self.target_w), mode="nearest")
        q = self.up1(q)
        q = self.up2(q)
        return torch.sigmoid(self.head(q))                               # [B, 3, H, W]


__all__ = ["BMPExpertNet"]
