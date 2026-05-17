"""SlotNetV3.1 -- V3 + GPT review patches.

Patches over V3:

1. **Input-conditioned default-atlas prior.** A pre-packed default-skin atlas
   gets a per-channel mean/std swap from the input view, and the model
   predicts a *residual* on top of that prior in logit space. Final RGB conv
   is zero-initialized so the prior is the starting point.

2. **Decoder layout conditioning.** At every decoder stage we inject
   atlas-space coord maps + valid mask + per-pixel slot loss weight +
   per-pixel slot-id embedding (downsampled to the stage's resolution).
   These tell the decoder *which* slot it is drawing -- no GeoNet routing.

3. The bottleneck and FiLM-style global style are kept from V3. We only
   add information; we don't remove anything.

Loss-side patches (lower hidden-pixel weight, sobel bump, contrastive loss)
live in `models/losses.py` + `models/atlas.py` + `train_slotnet.py`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

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


def color_transfer_default(default_atlas: torch.Tensor, input_view: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize the default atlas to match the input view's per-channel statistics.

    default_atlas: [3, H, W] or [B, 3, H, W]
    input_view:    [B, 3, Hv, Wv]
    Returns:       [B, 3, H, W]
    """
    if default_atlas.ndim == 3:
        default_atlas = default_atlas.unsqueeze(0).expand(input_view.shape[0], -1, -1, -1)
    src_mean = input_view.mean(dim=(2, 3), keepdim=True)
    src_std = input_view.std(dim=(2, 3), keepdim=True).clamp_min(eps)
    dst_mean = default_atlas.mean(dim=(2, 3), keepdim=True)
    dst_std = default_atlas.std(dim=(2, 3), keepdim=True).clamp_min(eps)
    return (((default_atlas - dst_mean) / dst_std) * src_std + src_mean).clamp(0, 1)


def build_layout_maps(atlas_profile, atlas_h: int, atlas_w: int, slot_emb_dim: int = 16) -> Dict[str, torch.Tensor]:
    """Build static atlas-space conditioning maps.

    Returns a dict with one tensor per channel set:
      xy        [2, H, W]   normalized x, y in [-1, 1]
      valid     [1, H, W]   1 where any slot lives, 0 otherwise
      weight    [1, H, W]   atlas_v1 loss_weight per pixel (no per-skin factor)
      slot_emb  [E, H, W]   per-pixel slot-id embedding (random init, learnable)
    """
    H, W = atlas_h, atlas_w
    ys = torch.linspace(-1.0, 1.0, H)
    xs = torch.linspace(-1.0, 1.0, W)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    xy = torch.stack((xx, yy), dim=0)  # [2, H, W]

    valid = torch.zeros((1, H, W))
    weight = torch.zeros((1, H, W))
    slot_id_map = torch.full((H, W), fill_value=-1, dtype=torch.long)
    for slot in atlas_profile.slots:
        y0, y1 = slot.y, slot.y + slot.h
        x0, x1 = slot.x, slot.x + slot.w
        valid[:, y0:y1, x0:x1] = 1.0
        weight[:, y0:y1, x0:x1] = float(slot.loss_weight)
        slot_id_map[y0:y1, x0:x1] = slot.id

    return {
        "xy": xy,
        "valid": valid,
        "weight": weight,
        "slot_id_map": slot_id_map,
    }


class StyledUpBlock(nn.Module):
    """Upsample x2, FiLM-modulate by style, concat layout channels, conv."""

    def __init__(self, in_ch: int, out_ch: int, groups: int, style_ch: int, layout_ch: int):
        super().__init__()
        self.style_proj = nn.Linear(style_ch, in_ch * 2)
        self.body = conv_block(in_ch + layout_ch, out_ch, groups)

    def forward(self, x: torch.Tensor, style: torch.Tensor, layout: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        gb = self.style_proj(style.to(x.dtype))
        gamma, beta = gb.chunk(2, dim=1)
        x = x * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        # Resize layout to x's resolution (it should already match per stage).
        if layout.shape[-2:] != x.shape[-2:]:
            layout = F.interpolate(layout, size=x.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat((x, layout), dim=1)
        return self.body(x)


class SlotNetV31(nn.Module):
    def __init__(
        self,
        atlas_profile,
        default_atlas: torch.Tensor,  # [3, H, W]
        base_channels: int = 24,
        slot_emb_dim: int = 16,
    ):
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

        # Layout: xy(2) + valid(1) + weight(1) + slot_emb(E) = 4 + E channels
        layout_ch = 2 + 1 + 1 + slot_emb_dim
        self.slot_embedding = nn.Embedding(self.num_slots + 1, slot_emb_dim)  # +1 for "outside" id

        # Resolutions: seed 32 -> 64 -> 128 -> 256 -> 512 -> 1024
        d0 = c5
        d1 = c4
        d2 = c3
        d3 = c2
        d4 = base_channels
        d5 = base_channels
        self.up1 = StyledUpBlock(d0, d1, 16, style_ch=128, layout_ch=layout_ch)
        self.up2 = StyledUpBlock(d1, d2, 16, style_ch=128, layout_ch=layout_ch)
        self.up3 = StyledUpBlock(d2, d3, 8, style_ch=128, layout_ch=layout_ch)
        self.up4 = StyledUpBlock(d3, d4, 8, style_ch=128, layout_ch=layout_ch)
        self.up5 = StyledUpBlock(d4, d5, 8, style_ch=128, layout_ch=layout_ch)

        self.out_head = nn.Conv2d(d5, 7, 1)
        # Zero-init RGB residual channels so prediction starts at the prior.
        with torch.no_grad():
            self.out_head.weight[:3].zero_()
            self.out_head.bias[:3].zero_()

        layout = build_layout_maps(atlas_profile, self.atlas_h, self.atlas_w, slot_emb_dim)
        self.register_buffer("layout_xy", layout["xy"])
        self.register_buffer("layout_valid", layout["valid"])
        w = layout["weight"]
        self.register_buffer("layout_weight", w / max(1.0, float(w.max())))
        self.register_buffer("layout_slot_id", layout["slot_id_map"].long())
        self.register_buffer("default_atlas", default_atlas.clamp(0.0, 1.0))

    def _layout_at(self, res: int) -> torch.Tensor:
        """Build the per-stage atlas layout tensor [4 + E, res, res] fresh per
        forward. The static maps (xy, valid, weight) are buffer-backed and the
        slot-id embedding is learnable, so we recompute and concat each step.
        Cheap because the only real work is the embedding lookup + a couple of
        interpolate calls on small tensors."""
        target = (res, res)
        xy = F.interpolate(self.layout_xy.unsqueeze(0), size=target, mode="bilinear", align_corners=False)[0]
        valid = F.interpolate(self.layout_valid.unsqueeze(0), size=target, mode="bilinear", align_corners=False)[0]
        weight = F.interpolate(self.layout_weight.unsqueeze(0), size=target, mode="bilinear", align_corners=False)[0]
        emb_full = self.slot_embedding((self.layout_slot_id + 1).clamp_min(0))  # [H, W, E]
        emb_full = emb_full.permute(2, 0, 1).unsqueeze(0)  # [1, E, H, W]
        emb = F.interpolate(emb_full, size=target, mode="bilinear", align_corners=False)[0]
        return torch.cat((xy, valid, weight, emb), dim=0)

    def encode(self, view: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f1 = self.enc1(view)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        f4 = self.enc4(f3)
        f5 = self.enc5(f4)
        style = self.style_proj(f5.mean(dim=(2, 3)))
        return f5, style

    def decode_residual(self, f5: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        seed = F.adaptive_avg_pool2d(f5, (32, 32))
        seed = self.seed_proj(seed)
        B = seed.shape[0]

        def lay(res):
            m = self._layout_at(res).to(seed.dtype)
            return m.unsqueeze(0).expand(B, -1, -1, -1)

        x = self.up1(seed, style, lay(64))
        x = self.up2(x, style, lay(128))
        x = self.up3(x, style, lay(256))
        x = self.up4(x, style, lay(512))
        x = self.up5(x, style, lay(1024))
        x = self.out_head(x)
        if x.shape[-2:] != (self.atlas_h, self.atlas_w):
            x = F.interpolate(x, size=(self.atlas_h, self.atlas_w), mode="bilinear", align_corners=False)
        return x

    def forward(self, view: torch.Tensor, residual_alpha: float = 1.0) -> dict[str, torch.Tensor]:
        f5, style = self.encode(view)
        residual_logits = self.decode_residual(f5, style)
        prior_rgb = color_transfer_default(self.default_atlas, view)
        prior_logit = torch.logit(prior_rgb.clamp(0.005, 0.995))
        # Scaled residual: residual_alpha modulates how strongly the residual
        # is allowed to deviate from the colorized-default prior. Used as a
        # warm-start schedule (small alpha early, ramping up) to prevent the
        # decoder from immediately smashing the prior structure.
        scaled_residual_rgb = residual_alpha * residual_logits[:, :3]
        rgb_logits = prior_logit + scaled_residual_rgb
        special_logits = residual_logits[:, 3:]
        prediction = torch.cat((rgb_logits, special_logits), dim=1)
        return {
            "prediction": prediction,
            "prior_rgb": prior_rgb,
            "residual_logits": residual_logits,
            "scaled_residual_rgb": scaled_residual_rgb,
        }


__all__ = [
    "SlotNetV31",
    "color_transfer_default",
    "build_layout_maps",
]
