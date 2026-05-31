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
                 heads: int = 4, attn_layers: int = 2, query_div: int = 4,
                 decoder_kind: str = "legacy", kv_scale: int = 1,
                 style_mod: bool = False, encoder: str = "scratch",
                 freqs: tuple[int, ...] = DEFAULT_FREQS,
                 kv_pool: tuple[tuple[int, int], ...] = DEFAULT_KV_POOL):
        super().__init__()
        self.target_h = int(target_h)
        self.target_w = int(target_w)
        self.attn_dim = attn_dim
        # Query grid is target H/query_div x W/query_div. Smaller divisor = finer
        # grid = more queries (more attention cost) but less reliance on the
        # final upsample, which large/detailed targets (e.g. EQMAIN 275x315) need
        # to break the ~mae 0.018 plateau seen with query_div=4.
        self.query_div = max(1, int(query_div))
        self.freqs = tuple(int(f) for f in freqs)
        # kv_scale multiplies the per-level K/V pool resolution. The default
        # avg-pool to ~2.4k tokens (kv_scale=1) is a ~700x averaging bottleneck
        # that blurs local appearance -> the decoder memorizes per-skin instead of
        # reading input detail (V11 diagnosis). kv_scale>1 gives the cross-attention
        # richer, less-lossy conditioning so it can generalize style->atlas.
        self.kv_scale = max(1, int(kv_scale))
        self.kv_pool = tuple((int(h) * self.kv_scale, int(w) * self.kv_scale)
                             for (h, w) in kv_pool)
        if len(self.kv_pool) != 4:
            raise ValueError("kv_pool must have one (h, w) per encoder level (4 total)")

        # Encoder. "scratch" = the from-scratch 4-stage CNN (memorizes with few
        # skins). "convnext" = FROZEN pretrained ConvNeXt-Tiny feature pyramid for
        # transferable visual features + a shallow raw-RGB stream (frozen nets drop
        # exact palette / 1px UI detail) — codex's top generalization lever.
        self.encoder = encoder
        if encoder == "convnext":
            from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
            bb = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT).features
            for p in bb.parameters():
                p.requires_grad_(False)
            self.cnx = bb.eval()
            self._cnx_capture = {1, 3, 5, 7}      # stage outputs: /4,/8,/16,/32
            self.register_buffer("imnet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("imnet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
            self.proj = nn.ModuleList([nn.Conv2d(c, attn_dim, 1) for c in (96, 192, 384, 768)])
            self.rgb_stem = nn.Sequential(_conv_block(3, 32, stride=2), _conv_block(32, 64, stride=2))
            self.rgb_proj = nn.Conv2d(64, attn_dim, 1)
        else:
            c1, c2, c3, c4 = base, base * 2, base * 4, base * 8
            self.stem = _conv_block(3, c1, stride=2)         # /2
            self.e1 = _conv_block(c1, c2, stride=2)          # /4
            self.e2 = _conv_block(c2, c3, stride=2)          # /8
            self.e3 = _conv_block(c3, c4, stride=2)          # /16
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

        # Decoder. "legacy": single nearest jump (query grid -> target) + 2 resblocks
        # (fine for smooth/small BMPs). "progressive": refine at half-res then
        # full-res before the 2 blocks, recovering high-frequency detail that the
        # single jump smears (needed for EQMAIN's dense slider sprite rows, where
        # legacy plateaus at ~mae 0.018 on rows 116-315). pre1/pre2 exist only in
        # progressive mode so legacy checkpoints load unchanged.
        if decoder_kind not in ("legacy", "progressive"):
            raise ValueError(f"decoder_kind must be legacy|progressive, got {decoder_kind!r}")
        self.decoder_kind = decoder_kind
        self.dec_proj = nn.Conv2d(attn_dim, dec_ch, 1)
        if decoder_kind == "progressive":
            self.pre1 = _ResBlock(dec_ch, dec_ch)   # refine at half target res
            self.pre2 = _ResBlock(dec_ch, dec_ch)   # refine at full target res
        self.up1 = _ResBlock(dec_ch, dec_ch)
        self.up2 = _ResBlock(dec_ch, dec_ch)
        self.head = nn.Conv2d(dec_ch, 3, 1)

        # V11 style modulation: factorize SHARED structure x per-skin STYLE.
        # struct_tokens = a learned, skin-independent content prior per atlas query
        # position (the "what goes where"); the global style code (from the deepest
        # encoder features) FiLM-modulates the decoder (skin art injected by
        # modulation, not generated from scratch). Off by default (baseline ckpts
        # load unchanged).
        self.style_mod = bool(style_mod)
        if self.style_mod:
            qh = max(1, self.target_h // self.query_div)
            qw = max(1, self.target_w // self.query_div)
            self.struct_tokens = nn.Parameter(torch.zeros(1, qh * qw, attn_dim))
            self.style_mlp = nn.Sequential(
                nn.Linear(base * 8, attn_dim), nn.SiLU(inplace=True),
                nn.Linear(attn_dim, dec_ch * 2))

        # Buffers for reconstructibility from a checkpoint alone.
        self.register_buffer("model_version", torch.tensor([10], dtype=torch.int32))
        self.register_buffer("target_h_buf", torch.tensor([self.target_h], dtype=torch.int32))
        self.register_buffer("target_w_buf", torch.tensor([self.target_w], dtype=torch.int32))
        for name, val in (("base_buf", base), ("attn_dim_buf", attn_dim),
                          ("dec_ch_buf", dec_ch), ("heads_buf", heads),
                          ("attn_layers_buf", attn_layers),
                          ("query_div_buf", self.query_div),
                          ("kv_scale_buf", self.kv_scale),
                          ("style_mod_buf", 1 if self.style_mod else 0),
                          ("encoder_buf", 1 if encoder == "convnext" else 0),
                          ("decoder_kind_buf", 1 if decoder_kind == "progressive" else 0)):
            self.register_buffer(name, torch.tensor([val], dtype=torch.int32))

    def _query_grid(self, b: int, device, dtype) -> torch.Tensor:
        qh = max(1, self.target_h // self.query_div)
        qw = max(1, self.target_w // self.query_div)
        coords = _fourier_coords(qh, qw, self.freqs, device, dtype)  # [C, qh, qw]
        coords = coords.permute(1, 2, 0).reshape(qh * qw, -1)         # [qh*qw, C]
        q = self.query_proj(coords).unsqueeze(0).expand(b, -1, -1)    # [B, qh*qw, attn_dim]
        return q, qh, qw

    def forward(self, x: torch.Tensor, return_style: bool = False):
        if x.dim() != 4 or x.shape[1] != 3:
            raise ValueError(f"x must be [B,3,H,W], got {tuple(x.shape)}")
        b = x.shape[0]
        if self.encoder == "convnext":
            self.cnx.eval()  # keep frozen backbone in eval even under model.train()
                             # (else ConvNeXt stochastic-depth reactivates — codex bug)
            with torch.no_grad():                      # frozen backbone
                h = (x - self.imnet_mean) / self.imnet_std
                stages = []
                for i, layer in enumerate(self.cnx):
                    h = layer(h)
                    if i in self._cnx_capture:
                        stages.append(h.float())
            feats = []
            for i, f in enumerate(stages):
                f = self.proj[i](f)
                feats.append(F.adaptive_avg_pool2d(f, self.kv_pool[i]))
            r = self.rgb_proj(self.rgb_stem(x))         # learnable raw-RGB stream
            feats.append(F.adaptive_avg_pool2d(r, self.kv_pool[0]))
            f4 = stages[-1]                              # for style_mod global code
        else:
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
        z_style = feats[3].mean(dim=(2, 3))   # [B, attn_dim] render style code (for cond-D)
        q, qh, qw = self._query_grid(b, x.device, x.dtype)
        if self.style_mod:
            q = q + self.struct_tokens                                   # shared structure prior
        for blk in self.attn_blocks:
            q = blk(q, kv)
        q = q.transpose(1, 2).reshape(b, self.attn_dim, qh, qw)         # [B, D, qh, qw]
        q = self.dec_proj(q)                                             # [B, dec_ch, qh, qw]
        if self.style_mod:
            style = f4.mean(dim=(2, 3))                                  # [B, c4] global style
            gamma, beta = self.style_mlp(style).chunk(2, dim=1)         # FiLM params
            q = q * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        if self.decoder_kind == "progressive":
            h2 = max(1, self.target_h // 2)
            w2 = max(1, self.target_w // 2)
            q = self.pre1(F.interpolate(q, size=(h2, w2), mode="nearest"))
            q = self.pre2(F.interpolate(q, size=(self.target_h, self.target_w), mode="nearest"))
        else:
            q = F.interpolate(q, size=(self.target_h, self.target_w), mode="nearest")
        q = self.up1(q)
        q = self.up2(q)
        out = torch.sigmoid(self.head(q))                                # [B, 3, H, W]
        return (out, z_style) if return_style else out


class BMPPatchDiscriminator(nn.Module):
    """PatchGAN discriminator for adversarial BMP-expert training.

    Judges real target BMP vs generated BMP at the patch level (a small
    receptive field per output cell), forcing the generator off the L1
    conditional-mean (blur) and onto crisp high-frequency detail — the part a
    pixel regressor cannot "imagine" (e.g. EQMAIN's 1px slider grooves).
    Spectral norm + LeakyReLU for stability; returns a logit map (hinge loss).
    Training-only: never part of the saved expert checkpoint, so inference /
    packaging / composition are unchanged.

    Returns (logits, features) — features feed an optional feature-matching loss.
    """

    def __init__(self, base: int = 64, n_layers: int = 3, min_dim: int | None = None,
                 style_dim: int | None = None):
        super().__init__()
        # Auto-cap stride-2 downsamples so the smaller target dim stays workable
        # (thin/small BMPs). Only the k4-s2 layers shrink; the final conv + head
        # are k3-s1 (size-preserving) so they never under-run the kernel.
        if min_dim is not None:
            cap = max(1, int(math.floor(math.log2(max(2, min_dim)))) - 1)
            n_layers = max(1, min(n_layers, cap))
        def sn(c_in, c_out, k, stride):
            return nn.utils.spectral_norm(
                nn.Conv2d(c_in, c_out, kernel_size=k, stride=stride, padding=k // 2))
        layers = [sn(3, base, 4, 2), nn.LeakyReLU(0.2, inplace=True)]
        ch = base
        for _ in range(1, n_layers):
            nxt = min(ch * 2, base * 8)
            layers += [sn(ch, nxt, 4, 2), nn.LeakyReLU(0.2, inplace=True)]
            ch = nxt
        nxt = min(ch * 2, base * 8)
        layers += [sn(ch, nxt, 3, 1), nn.LeakyReLU(0.2, inplace=True)]  # size-preserving
        self.body = nn.ModuleList(layers)
        self.head = sn(nxt, 1, 3, 1)
        # Conditional projection head (Miyato): scores "is this atlas patch
        # compatible with this render-style code z?" z comes from the generator's
        # input-render encoder (available at product inference). Off if style_dim None.
        self._final_ch = nxt
        if style_dim is not None:
            self.style_ln = nn.LayerNorm(style_dim)
            self.style_proj = nn.Linear(style_dim, nxt)

    def forward(self, x: torch.Tensor, z: torch.Tensor | None = None):
        feats = []
        h = x
        for m in self.body:
            h = m(h)
            if isinstance(m, nn.LeakyReLU):
                feats.append(h)
        logits = self.head(h)
        if z is not None and hasattr(self, "style_proj"):
            s = self.style_proj(self.style_ln(z))                     # [B, C]
            proj = (h * s[:, :, None, None]).sum(1, keepdim=True) / (self._final_ch ** 0.5)
            logits = logits + proj
        return logits, feats


__all__ = ["BMPExpertNet", "BMPPatchDiscriminator"]
