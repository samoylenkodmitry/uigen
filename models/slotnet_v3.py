"""SlotNetV3: full image -> full 1024x1024x7 atlas in one shot.

Design principles (correcting V1/V2):

- SlotNet input is **only the source image**. GeoNet output (rects, state) is
  pipeline-level provenance / debug, but does NOT participate in SlotNet's
  code path -- no ROI alignment, no per-slot routing, no cropping decisions
  derived from GeoNet.
- SlotNet output is the **whole packed atlas** at its native 1024x1024 RGB +
  4 special-class channels. Per-slot cropping happens only in
  `export_atlas_to_skin` (atlas -> BMP).
- The decoder gets a global style vector (pooled from the encoder) and a
  spatial bottleneck (encoder features adaptively pooled to 32x32). It
  upsamples to 1024x1024 with conditioning at every stage. Slot layout is
  learned implicitly from training data; the model produces the right slot
  in the right atlas position because that is what the atlas targets show.

The output channel layout matches V1/V2: channels 0-2 are RGB logits
(`sigmoid` applied in the loss / inference), channels 3-6 are special-class
logits used for magenta keying on slots whose policy enables it.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def conv_block(in_ch: int, out_ch: int, groups: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
        nn.GroupNorm(groups, out_ch),
        nn.SiLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.GroupNorm(groups, out_ch),
        nn.SiLU(inplace=True),
    )


def coord_channels(h: int, w: int, device, dtype) -> torch.Tensor:
    ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((xx, yy), dim=0)  # [2, H, W]


class StyledUpBlock(nn.Module):
    """Bilinear upsample x2, then conv block with global style conditioning."""

    def __init__(self, in_ch: int, out_ch: int, groups: int, style_ch: int):
        super().__init__()
        self.style_proj = nn.Linear(style_ch, in_ch * 2)  # gamma, beta for FiLM
        self.body = conv_block(in_ch + 2, out_ch, groups)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        # FiLM modulation on the input channels.
        gb = self.style_proj(style.to(x.dtype))
        gamma, beta = gb.chunk(2, dim=1)
        x = x * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        coords = coord_channels(x.shape[2], x.shape[3], x.device, x.dtype)
        coords = coords[None].expand(x.shape[0], -1, -1, -1)
        x = torch.cat((x, coords), dim=1)
        return self.body(x)


class SlotNetV3(nn.Module):
    def __init__(self, base_channels: int = 24, atlas_h: int = 1024, atlas_w: int = 1024):
        super().__init__()
        self.atlas_h = atlas_h
        self.atlas_w = atlas_w

        c1 = base_channels         # stride 1   1728x960
        c2 = base_channels * 2     # stride 2    864x480
        c3 = base_channels * 4     # stride 4    432x240
        c4 = base_channels * 6     # stride 8    216x120
        c5 = base_channels * 8     # stride 16   108x60

        self.enc1 = conv_block(3, c1, 8, stride=1)
        self.enc2 = conv_block(c1, c2, 8, stride=2)
        self.enc3 = conv_block(c2, c3, 16, stride=2)
        self.enc4 = conv_block(c3, c4, 16, stride=2)
        self.enc5 = conv_block(c4, c5, 32, stride=2)

        self.style_proj = nn.Linear(c5, 128)
        # Atlas decoder starts at 32x32. We adaptive-pool encoder features here.
        self.seed_proj = nn.Conv2d(c5, c5, 1)

        # Upsample chain: 32 -> 64 -> 128 -> 256 -> 512 -> 1024 (5 ups).
        d0 = c5            # 192 channels @ 32x32 start
        d1 = c4            # 144 @ 64x64
        d2 = c3            # 96 @ 128x128
        d3 = c2            # 48 @ 256x256
        d4 = base_channels # 24 @ 512x512
        d5 = base_channels # 24 @ 1024x1024

        self.up1 = StyledUpBlock(d0, d1, 16, style_ch=128)
        self.up2 = StyledUpBlock(d1, d2, 16, style_ch=128)
        self.up3 = StyledUpBlock(d2, d3, 8, style_ch=128)
        self.up4 = StyledUpBlock(d3, d4, 8, style_ch=128)
        self.up5 = StyledUpBlock(d4, d5, 8, style_ch=128)

        self.out_head = nn.Conv2d(d5, 7, 1)

    def encode(self, view: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f1 = self.enc1(view)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        f4 = self.enc4(f3)
        f5 = self.enc5(f4)
        style = self.style_proj(f5.mean(dim=(2, 3)))
        return f5, style

    def decode(self, f5: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        # Adapt encoder bottleneck to a 32x32 seed grid via adaptive avg pool.
        seed = F.adaptive_avg_pool2d(f5, (32, 32))
        seed = self.seed_proj(seed)
        x = self.up1(seed, style)
        x = self.up2(x, style)
        x = self.up3(x, style)
        x = self.up4(x, style)
        x = self.up5(x, style)
        return self.out_head(x)

    def forward(self, view: torch.Tensor) -> dict[str, torch.Tensor]:
        f5, style = self.encode(view)
        atlas_logits = self.decode(f5, style)
        # Pad/crop to atlas_h x atlas_w in case upsample chain doesn't land
        # exactly. With 32->1024 (5 ups x 2 each) we land at 1024 exactly.
        if atlas_logits.shape[-2] != self.atlas_h or atlas_logits.shape[-1] != self.atlas_w:
            atlas_logits = F.interpolate(
                atlas_logits, size=(self.atlas_h, self.atlas_w), mode="bilinear", align_corners=False
            )
        return {"prediction": atlas_logits}


__all__ = ["SlotNetV3"]
