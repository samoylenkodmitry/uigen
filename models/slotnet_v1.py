from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .crop import coordinate_channels, crop_view_regions


def block(in_ch: int, out_ch: int, groups: int = 8, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
        nn.GroupNorm(groups, out_ch),
        nn.SiLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.GroupNorm(groups, out_ch),
        nn.SiLU(inplace=True),
    )


class SlotNetV1(nn.Module):
    def __init__(self, slots: int = 18, states: int = 32, input_channels: int = 36):
        super().__init__()
        self.slot_embedding = nn.Embedding(slots, 16)
        self.state_mlp = nn.Sequential(nn.Linear(states, 64), nn.SiLU(inplace=True), nn.Linear(64, 8))

        self.enc0 = block(input_channels, 64, 8)
        self.down1 = block(64, 128, 8, stride=2)
        self.down2 = block(128, 256, 16, stride=2)
        self.down3 = block(256, 512, 32, stride=2)
        self.bottleneck = block(512, 512, 32)
        self.bottleneck256 = block(256, 256, 16)
        self.up2 = block(512 + 256, 256, 16)
        self.up1_deep = block(256 + 128, 128, 8)
        self.up1_shallow = block(256 + 128, 128, 8)
        self.up0 = block(128 + 64, 64, 8)
        self.out = nn.Conv2d(64, 7, 1)

    def build_conditioned_input(
        self,
        crop: torch.Tensor,
        slot_id: torch.Tensor,
        rect: torch.Tensor,
        state: torch.Tensor,
        log_scale: torch.Tensor,
    ) -> torch.Tensor:
        batch, _, height, width = crop.shape
        coords = coordinate_channels(batch, height, width, crop.device, crop.dtype)
        slot = self.slot_embedding(slot_id).to(crop.dtype)[:, :, None, None].expand(-1, -1, height, width)
        state_emb = self.state_mlp(state.to(crop.dtype))[:, :, None, None].expand(-1, -1, height, width)
        rect_ch = rect.to(crop.dtype)[:, :, None, None].expand(-1, -1, height, width)
        scale_ch = log_scale.to(crop.dtype)[:, :, None, None].expand(-1, -1, height, width)
        return torch.cat((crop, coords, slot, state_emb, rect_ch, scale_ch), dim=1)

    def forward_conditioned(self, x: torch.Tensor, shallow: bool = False) -> torch.Tensor:
        e0 = self.enc0(x)
        d1 = self.down1(e0)
        d2 = self.down2(d1)
        if shallow:
            b = self.bottleneck256(d2)
            u1 = F.interpolate(b, size=d1.shape[-2:], mode="bilinear", align_corners=False)
            u1 = self.up1_shallow(torch.cat((u1, d1), dim=1))
        else:
            d3 = self.down3(d2)
            b = self.bottleneck(d3)
            u2 = F.interpolate(b, size=d2.shape[-2:], mode="bilinear", align_corners=False)
            u2 = self.up2(torch.cat((u2, d2), dim=1))
            u1 = F.interpolate(u2, size=d1.shape[-2:], mode="bilinear", align_corners=False)
            u1 = self.up1_deep(torch.cat((u1, d1), dim=1))
        u0 = F.interpolate(u1, size=e0.shape[-2:], mode="bilinear", align_corners=False)
        u0 = self.up0(torch.cat((u0, e0), dim=1))
        return self.out(u0)

    def forward(
        self,
        view: torch.Tensor,
        source_rect: torch.Tensor,
        state: torch.Tensor,
        slot_id: torch.Tensor,
        output_hw: tuple[int, int],
        input_hw: tuple[int, int] = (1280, 768),
    ) -> dict[str, torch.Tensor]:
        crop, log_scale, valid = crop_view_regions(view, source_rect, output_hw, input_hw)
        conditioned = self.build_conditioned_input(crop, slot_id, source_rect, state, log_scale)
        shallow = min(output_hw) < 64
        return {
            "prediction": self.forward_conditioned(conditioned, shallow=shallow),
            "crop": crop,
            "log_scale": log_scale,
            "valid": valid,
        }
