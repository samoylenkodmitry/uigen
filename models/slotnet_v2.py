"""SlotNetV2: full-view shared encoder + per-slot ROI-aligned decoder.

Design rationale (correcting the original §23-31 simplification):

- The original SlotNetV1 cropped the input view to each slot's predicted source
  rect BEFORE running any conv layers. That discarded global style/palette
  context and made the model fragile to small GeoNet rect errors (which
  swallowed letterbox padding into the crop).

- V2 keeps the same output contract -- `[B, 7, slot_h, slot_w]` per slot -- so
  the rest of the pipeline (atlas paste, BMP export, .wsz zip) is unchanged.

- Internally V2 runs a shared encoder over the full input view to capture
  global palette/style, then for each slot:
    1. ROI-align encoder features (stride-4 and stride-16) at the slot's atlas
       dimensions. This is the "crop" -- but in feature space, after the
       encoder has already seen the whole image.
    2. Concatenate global pooled style + slot embedding + state embedding +
       rect channels + coord channels.
    3. Decode to `[B, 7, slot_h, slot_w]`.

For training and inference, the same model is called once per slot per sample
in the current code path. Memory: shared encoder caches activations once; the
per-slot decoders are small. Inference time is dominated by one encoder pass
per image, not by the per-slot decoders.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .crop import coordinate_channels


def conv_block(in_ch: int, out_ch: int, groups: int, stride: int = 1) -> nn.Sequential:
    """Two-conv + GroupNorm + SiLU block, optional stride on the first conv."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
        nn.GroupNorm(groups, out_ch),
        nn.SiLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.GroupNorm(groups, out_ch),
        nn.SiLU(inplace=True),
    )


def roi_align_features(
    feats: torch.Tensor,
    rects: torch.Tensor,
    output_hw: tuple[int, int],
) -> torch.Tensor:
    """Sample `feats` at normalized `rects` into `output_hw` resolution.

    `feats` is `[B, C, H, W]`. `rects` is `[B, 5]` with
    `[x0_norm, y0_norm, x1_norm, y1_norm, visible_flag]`. Returns
    `[B, C, out_h, out_w]`. Padding mode is `border` so that a rect
    overflowing the feature map extrapolates the edge rather than zeroing
    out (which used to bake letterbox into V1's crops).
    """
    out_h, out_w = output_hw
    device = feats.device
    dtype = feats.dtype
    ys = (torch.arange(out_h, device=device, dtype=dtype) + 0.5) / out_h
    xs = (torch.arange(out_w, device=device, dtype=dtype) + 0.5) / out_w
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    x0, y0, x1, y1, _ = rects.to(device=device, dtype=dtype).unbind(dim=1)
    width = (x1 - x0).clamp_min(1e-6)
    height = (y1 - y0).clamp_min(1e-6)

    grid_x = x0[:, None, None] + xx[None, :, :] * width[:, None, None]
    grid_y = y0[:, None, None] + yy[None, :, :] * height[:, None, None]
    grid = torch.stack((grid_x * 2.0 - 1.0, grid_y * 2.0 - 1.0), dim=-1)
    return F.grid_sample(feats, grid, mode="bilinear", padding_mode="border", align_corners=False)


class SlotNetV2(nn.Module):
    def __init__(self, slots: int = 18, states: int = 32, base_channels: int = 32):
        super().__init__()
        c1 = base_channels         # stride 1
        c2 = base_channels * 2     # stride 2
        c3 = base_channels * 4     # stride 4
        c4 = base_channels * 6     # stride 8
        c5 = base_channels * 8     # stride 16

        self.enc1 = conv_block(3, c1, 8, stride=1)
        self.enc2 = conv_block(c1, c2, 8, stride=2)
        self.enc3 = conv_block(c2, c3, 16, stride=2)
        self.enc4 = conv_block(c3, c4, 16, stride=2)
        self.enc5 = conv_block(c4, c5, 32, stride=2)

        self.slot_embedding = nn.Embedding(slots, 32)
        self.state_mlp = nn.Sequential(
            nn.Linear(states, 64), nn.SiLU(inplace=True), nn.Linear(64, 16)
        )

        # Decoder input channels:
        #   ROI of stride-4 features: c3
        #   ROI of stride-16 features: c5
        #   Global pooled style (from stride-16): c5
        #   Slot embedding: 32
        #   State embedding: 16
        #   Rect channels (x0, y0, x1, y1, visible): 5
        #   Coord channels: 2
        dec_in = c3 + c5 + c5 + 32 + 16 + 5 + 2
        self.dec_proj = nn.Conv2d(dec_in, 128, 1)
        self.dec = nn.Sequential(
            conv_block(128, 128, 8),
            conv_block(128, 96, 8),
            conv_block(96, 64, 8),
        )
        self.out_head = nn.Conv2d(64, 7, 1)

    def encode(self, view: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (stride-4 features, stride-16 features, global pooled [B, c5])."""
        f1 = self.enc1(view)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        f4 = self.enc4(f3)
        f5 = self.enc5(f4)
        gp = f5.mean(dim=(2, 3))
        return f3, f5, gp

    def decode_slot(
        self,
        f3: torch.Tensor,
        f5: torch.Tensor,
        gp: torch.Tensor,
        source_rect: torch.Tensor,
        state: torch.Tensor,
        slot_id: torch.Tensor,
        output_hw: tuple[int, int],
        view: torch.Tensor | None = None,
    ) -> torch.Tensor:
        oh, ow = output_hw
        roi_lo = roi_align_features(f3, source_rect, output_hw)
        roi_hi = roi_align_features(f5, source_rect, output_hw)

        B = f3.shape[0]
        dtype = f3.dtype
        device = f3.device
        slot_emb = self.slot_embedding(slot_id).to(dtype)[:, :, None, None].expand(-1, -1, oh, ow)
        state_emb = self.state_mlp(state.to(dtype))[:, :, None, None].expand(-1, -1, oh, ow)
        rect_ch = source_rect.to(dtype)[:, :, None, None].expand(-1, -1, oh, ow)
        global_ch = gp.to(dtype)[:, :, None, None].expand(-1, -1, oh, ow)
        coord = coordinate_channels(B, oh, ow, device, dtype)

        x = torch.cat((roi_lo, roi_hi, global_ch, slot_emb, state_emb, rect_ch, coord), dim=1)
        x = self.dec_proj(x)
        x = self.dec(x)
        decoded = self.out_head(x)

        # Identity-skip on RGB logits: add inv_sigmoid(view_roi) so the simplest
        # valid prediction is "reproduce the input region as-is." The decoder
        # only has to learn the residual (recolor / denoise / hidden-state
        # recovery). This blocks the mean-color collapse that degenerates from
        # mask-normalized L1 with shared slot heads.
        if view is not None:
            view_roi = roi_align_features(view, source_rect, output_hw)
            identity_logit = torch.logit(view_roi.clamp(1e-3, 1.0 - 1e-3))
            decoded = decoded.clone()
            decoded[:, :3] = decoded[:, :3] + identity_logit
        return decoded

    def forward(
        self,
        view: torch.Tensor,
        source_rect: torch.Tensor,
        state: torch.Tensor,
        slot_id: torch.Tensor,
        output_hw: tuple[int, int],
        input_hw: tuple[int, int] = (1728, 960),
    ) -> dict[str, torch.Tensor]:
        f3, f5, gp = self.encode(view)
        prediction = self.decode_slot(f3, f5, gp, source_rect, state, slot_id, output_hw, view=view)
        valid = (source_rect[:, 4] > 0.0) & ((source_rect[:, 2] - source_rect[:, 0]) > 0.0) & ((source_rect[:, 3] - source_rect[:, 1]) > 0.0)
        # `crop` and `log_scale` are emitted for compatibility with V1 callers
        # (train_slotnet / infer_skin / debug grids). `crop` here is the
        # roi-aligned low-stride features collapsed to 3 channels for display.
        # log_scale is zero because V2 does not depend on per-slot scaling.
        crop_proxy = roi_align_features(view, source_rect, output_hw)
        log_scale = torch.zeros(view.shape[0], 2, device=view.device, dtype=view.dtype)
        return {
            "prediction": prediction,
            "crop": crop_proxy,
            "log_scale": log_scale,
            "valid": valid,
        }


__all__ = ["SlotNetV2", "roi_align_features"]
