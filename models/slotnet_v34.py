"""SlotNetV3.4: direct image-to-RGB-atlas model.

The training contract is intentionally small:

    input rendered PNG -> predicted RGB atlas PNG -> expected RGB atlas PNG

Unsupported Cranamp components are not represented in the training atlas.
There is no default-skin prior, observed auxiliary head, special-color head,
dynamic mask, or per-slot loss weight.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


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


@dataclass(frozen=True)
class LayoutMaps:
    xy: torch.Tensor
    slot_id_map: torch.Tensor


def build_layout_maps(atlas_profile, atlas_h: int, atlas_w: int) -> LayoutMaps:
    ys = torch.linspace(-1.0, 1.0, atlas_h)
    xs = torch.linspace(-1.0, 1.0, atlas_w)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    xy = torch.stack((xx, yy), dim=0)
    slot_id_map = torch.full((atlas_h, atlas_w), fill_value=-1, dtype=torch.long)
    for slot in atlas_profile.slots:
        slot_id_map[slot.y:slot.y + slot.h, slot.x:slot.x + slot.w] = slot.id
    return LayoutMaps(xy=xy, slot_id_map=slot_id_map)


class StyledUpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, preferred_groups: int, style_ch: int, layout_ch: int):
        super().__init__()
        self.style_proj = nn.Linear(style_ch, in_ch * 2)
        self.body = conv_block(in_ch + layout_ch, out_ch, preferred_groups)

    def forward(self, x: torch.Tensor, style: torch.Tensor, layout: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        gamma, beta = self.style_proj(style.to(x.dtype)).chunk(2, dim=1)
        x = x * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        if layout.shape[-2:] != x.shape[-2:]:
            layout = F.interpolate(layout, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.body(torch.cat((x, layout), dim=1))


class SlotNetV34(nn.Module):
    def __init__(self, atlas_profile, base_channels: int = 24, slot_emb_dim: int = 16):
        super().__init__()
        self.atlas_h = atlas_profile.canvas_h
        self.atlas_w = atlas_profile.canvas_w
        self.num_slots = max(slot.id for slot in atlas_profile.slots) + 1
        self.slot_emb_dim = slot_emb_dim

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
        self.style_proj = nn.Linear(c5, 128)
        self.seed_proj = nn.Conv2d(c5, c5, 1)

        layout_ch = 2 + slot_emb_dim
        self.slot_embedding = nn.Embedding(self.num_slots + 1, slot_emb_dim)
        self.up1 = StyledUpBlock(c5, c4, 16, style_ch=128, layout_ch=layout_ch)
        self.up2 = StyledUpBlock(c4, c3, 16, style_ch=128, layout_ch=layout_ch)
        self.up3 = StyledUpBlock(c3, c2, 8, style_ch=128, layout_ch=layout_ch)
        self.up4 = StyledUpBlock(c2, c1, 8, style_ch=128, layout_ch=layout_ch)
        self.up5 = StyledUpBlock(c1, c1, 8, style_ch=128, layout_ch=layout_ch)
        self.out_head = nn.Conv2d(c1, 3, 1)
        nn.init.kaiming_normal_(self.out_head.weight)
        nn.init.zeros_(self.out_head.bias)

        layout = build_layout_maps(atlas_profile, self.atlas_h, self.atlas_w)
        self.register_buffer("layout_xy", layout.xy)
        self.register_buffer("layout_slot_id", layout.slot_id_map.long())
        self.register_buffer("slotnet_version", torch.tensor([34], dtype=torch.int32))

    def _layout_at(self, res: int) -> torch.Tensor:
        target = (res, res)
        xy = F.interpolate(self.layout_xy.unsqueeze(0), size=target, mode="bilinear", align_corners=False)[0]
        slot_ids = (self.layout_slot_id + 1).clamp_min(0)
        emb_full = self.slot_embedding(slot_ids).permute(2, 0, 1).unsqueeze(0)
        emb = F.interpolate(emb_full, size=target, mode="bilinear", align_corners=False)[0]
        return torch.cat((xy, emb), dim=0)

    def encode(self, view: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f1 = self.enc1(view)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        f4 = self.enc4(f3)
        f5 = self.enc5(f4)
        style = self.style_proj(f5.mean(dim=(2, 3)))
        return f5, style

    def decode(self, f5: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        seed = self.seed_proj(F.adaptive_avg_pool2d(f5, (32, 32)))
        batch = seed.shape[0]

        def lay(res: int) -> torch.Tensor:
            layout = self._layout_at(res).to(seed.dtype)
            return layout.unsqueeze(0).expand(batch, -1, -1, -1)

        x = self.up1(seed, style, lay(64))
        x = self.up2(x, style, lay(128))
        x = self.up3(x, style, lay(256))
        x = self.up4(x, style, lay(512))
        x = self.up5(x, style, lay(1024))
        x = self.out_head(x)
        if x.shape[-2:] != (self.atlas_h, self.atlas_w):
            x = F.interpolate(x, size=(self.atlas_h, self.atlas_w), mode="bilinear", align_corners=False)
        return x

    def forward(self, view: torch.Tensor) -> dict[str, torch.Tensor]:
        f5, style = self.encode(view)
        return {"prediction": self.decode(f5, style)}


__all__ = ["SlotNetV34", "build_layout_maps"]
