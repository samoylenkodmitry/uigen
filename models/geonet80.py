from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from atlas_ai.rects import derive_eq_band_rects
from .losses import centernet_focal_loss


STATE_ANCHORS = [
    6, 13, 15, 5, 17, 18, 27, 27, 28,
    30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
    46, 44, 0, 0, 24, 24, 42, 42, 0, 0, 0, 0, 0,
]


def conv_block(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(min(8, out_ch), out_ch),
        nn.SiLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.GroupNorm(min(8, out_ch), out_ch),
        nn.SiLU(inplace=True),
    )


class GeoNet80(nn.Module):
    """Smoke-capable fixed-class detector for Cranamp render geometry."""

    def __init__(self, components: int = 80, states: int = 32, fpn_channels: int = 128, base_channels: int = 32):
        super().__init__()
        self.components = components
        self.states = states
        c2, c3, c4, c5 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, 7, stride=2, padding=3, bias=False),
            nn.GroupNorm(min(8, base_channels), base_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(min(8, base_channels), base_channels),
            nn.SiLU(inplace=True),
        )
        self.c2 = conv_block(base_channels, c2)
        self.c3 = conv_block(c2, c3, stride=2)
        self.c4 = conv_block(c3, c4, stride=2)
        self.c5 = conv_block(c4, c5, stride=2)
        self.lat2 = nn.Conv2d(c2, fpn_channels, 1)
        self.lat3 = nn.Conv2d(c3, fpn_channels, 1)
        self.lat4 = nn.Conv2d(c4, fpn_channels, 1)
        self.lat5 = nn.Conv2d(c5, fpn_channels, 1)
        self.fpn = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.heatmap = nn.Sequential(nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(fpn_channels, components, 1))
        self.wh = nn.Sequential(nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(fpn_channels, components * 2, 1), nn.Softplus())
        self.offset = nn.Sequential(nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(fpn_channels, components * 2, 1), nn.Tanh())
        self.state_embedding = nn.Embedding(states, 16)
        self.state_mlp = nn.Sequential(nn.Linear(fpn_channels + 16 + 5, 64), nn.ReLU(inplace=True), nn.Linear(64, 1), nn.Sigmoid())

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        c2 = self.c2(x)
        c3 = self.c3(c2)
        c4 = self.c4(c3)
        c5 = self.c5(c4)
        size = c2.shape[-2:]
        p = self.lat2(c2)
        for lat, c in [(self.lat3, c3), (self.lat4, c4), (self.lat5, c5)]:
            p = p + F.interpolate(lat(c), size=size, mode="nearest")
        return self.fpn(p)

    def detection_heads(self, p: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "heatmap": self.heatmap(p).sigmoid(),
            "wh": self.wh(p),
            "offset": self.offset(p),
            "features": p,
        }

    def forward(self, x: torch.Tensor, anchor_rects: torch.Tensor | None = None, jitter_state_anchors: bool = False) -> dict[str, torch.Tensor]:
        p = self.features(x)
        out = self.detection_heads(p)
        if anchor_rects is None:
            anchor_rects = decode_rects(out["heatmap"], out["wh"], out["offset"])
        out["state"] = self.decode_state(p, anchor_rects, jitter_state_anchors)
        return out

    def decode_state(self, p: torch.Tensor, rects: torch.Tensor, jitter: bool = False) -> torch.Tensor:
        sampled = sample_state_features(p, rects, jitter=jitter)
        batch = p.shape[0]
        state_ids = torch.arange(self.states, device=p.device)
        emb = self.state_embedding(state_ids).unsqueeze(0).expand(batch, -1, -1)
        anchor_rects = rects[:, STATE_ANCHORS, :].to(p.dtype)
        mlp_in = torch.cat((sampled, emb, anchor_rects), dim=-1)
        return self.state_mlp(mlp_in).squeeze(-1)


def sample_state_features(p: torch.Tensor, rects: torch.Tensor, jitter: bool = False) -> torch.Tensor:
    batch, channels, height, width = p.shape
    padded = F.pad(p, (1, 1, 1, 1), mode="replicate")
    outputs = []
    for state_idx, comp_id in enumerate(STATE_ANCHORS):
        rect = rects[:, comp_id, :].to(p.dtype)
        x0, y0, x1, y1, visible = rect.unbind(dim=1)
        cx = (x0 + x1) * 0.5 * width
        cy = (y0 + y1) * 0.5 * height
        if jitter:
            w_cells = (x1 - x0).clamp_min(1e-6) * width
            h_cells = (y1 - y0).clamp_min(1e-6) * height
            jx = (torch.rand_like(cx) * 2.0 - 1.0) * torch.minimum(torch.full_like(w_cells, 2.0), 0.25 * w_cells)
            jy = (torch.rand_like(cy) * 2.0 - 1.0) * torch.minimum(torch.full_like(h_cells, 2.0), 0.25 * h_cells)
            cx = cx + jx
            cy = cy + jy
        ix = cx.round().clamp(0, width - 1).long() + 1
        iy = cy.round().clamp(0, height - 1).long() + 1
        values = []
        for b in range(batch):
            if visible[b] <= 0:
                values.append(torch.zeros(channels, device=p.device, dtype=p.dtype))
            else:
                patch = padded[b, :, iy[b] - 1 : iy[b] + 2, ix[b] - 1 : ix[b] + 2]
                values.append(patch.mean(dim=(1, 2)))
        outputs.append(torch.stack(values, dim=0))
    return torch.stack(outputs, dim=1)


def build_geonet_targets(
    rects: torch.Tensor,
    grid_hw: tuple[int, int],
    components: int = 80,
    train_eq_band_heatmaps: bool = False,
) -> dict[str, torch.Tensor]:
    batch = rects.shape[0]
    grid_h, grid_w = grid_hw
    heatmap = torch.zeros((batch, components, grid_h, grid_w), dtype=rects.dtype, device=rects.device)
    wh = torch.zeros((batch, components, 2, grid_h, grid_w), dtype=rects.dtype, device=rects.device)
    offset = torch.zeros_like(wh)
    reg_mask = torch.zeros((batch, components, grid_h, grid_w), dtype=torch.bool, device=rects.device)
    valid_heatmap = torch.ones_like(heatmap, dtype=torch.bool)
    valid_heatmap[:, 60:80] = False
    if not train_eq_band_heatmaps:
        valid_heatmap[:, 30:40] = False

    for b in range(batch):
        for k in range(components):
            if not valid_heatmap[b, k, 0, 0] or rects[b, k, 4] <= 0:
                continue
            x0, y0, x1, y1, _ = rects[b, k]
            cx = (x0 + x1) * 0.5 * grid_w
            cy = (y0 + y1) * 0.5 * grid_h
            gx = int(torch.floor(cx).clamp(0, grid_w - 1).item())
            gy = int(torch.floor(cy).clamp(0, grid_h - 1).item())
            for yy in range(max(0, gy - 2), min(grid_h, gy + 3)):
                for xx in range(max(0, gx - 2), min(grid_w, gx + 3)):
                    dist2 = (xx - cx).pow(2) + (yy - cy).pow(2)
                    heatmap[b, k, yy, xx] = torch.maximum(heatmap[b, k, yy, xx], torch.exp(-dist2 / 2.0))
            wh[b, k, :, gy, gx] = torch.stack((x1 - x0, y1 - y0))
            offset[b, k, :, gy, gx] = torch.stack((cx - torch.floor(cx), cy - torch.floor(cy)))
            reg_mask[b, k, gy, gx] = True
    return {"heatmap": heatmap, "wh": wh.flatten(1, 2), "offset": offset.flatten(1, 2), "reg_mask": reg_mask, "valid_heatmap": valid_heatmap}


def geonet_loss(outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor], state_target: torch.Tensor) -> dict[str, torch.Tensor]:
    center = centernet_focal_loss(outputs["heatmap"], targets["heatmap"], targets["valid_heatmap"], positives=targets["reg_mask"])
    mask = targets["reg_mask"].repeat_interleave(2, dim=1)
    wh = F.smooth_l1_loss(outputs["wh"][mask], targets["wh"][mask]) if mask.any() else outputs["wh"].sum() * 0.0
    off = F.smooth_l1_loss(outputs["offset"][mask], targets["offset"][mask]) if mask.any() else outputs["offset"].sum() * 0.0
    state = F.smooth_l1_loss(outputs["state"], state_target)
    total = center + 5.0 * wh + 2.0 * off + state
    return {"total": total, "center": center, "wh": wh, "offset": off, "state": state}


def decode_rects(heatmap: torch.Tensor, wh: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    batch, components, grid_h, grid_w = heatmap.shape
    wh = wh.view(batch, components, 2, grid_h, grid_w)
    offset = offset.view(batch, components, 2, grid_h, grid_w)
    rects = torch.zeros((batch, components, 5), device=heatmap.device, dtype=heatmap.dtype)
    flat = heatmap.flatten(2)
    scores, indices = flat.max(dim=2)
    gy = torch.div(indices, grid_w, rounding_mode="floor")
    gx = indices % grid_w
    for b in range(batch):
        for k in range(components):
            x = gx[b, k]
            y = gy[b, k]
            ow, oh = wh[b, k, :, y, x]
            dx, dy = offset[b, k, :, y, x]
            cx = (x.to(heatmap.dtype) + dx) / grid_w
            cy = (y.to(heatmap.dtype) + dy) / grid_h
            rects[b, k] = torch.stack(((cx - ow * 0.5).clamp(0, 1), (cy - oh * 0.5).clamp(0, 1), (cx + ow * 0.5).clamp(0, 1), (cy + oh * 0.5).clamp(0, 1), (scores[b, k] > 0.25).to(heatmap.dtype)))
        for idx, band in enumerate(derive_eq_band_rects(tuple(rects[b, 29].detach().cpu().tolist()))):
            rects[b, 30 + idx] = torch.tensor(band, device=heatmap.device, dtype=heatmap.dtype)
    return rects
