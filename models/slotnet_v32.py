"""SlotNetV3.2 -- V3.1 + residual gating on dead slots + observed-atlas aux head.

Two changes over V3.1:

1. Residual-enabled mask. The residual RGB is gated by a [1, H, W] mask that
   is 1 only where atlas_v1 has `loss_weight > 0`. NUMBERS / TEXT / GEN /
   VIDEO / RESERVED never receive gradient anyway, so we don't want the
   residual to add noise there -- those regions stay exactly at the
   color-transferred default prior.

2. Auxiliary observed-atlas head. The decoder produces a second 4-channel
   output: 3 RGB + 1 mask logit. The training target is the *visible
   provenance* atlas:
       observed_target_rgb  = target_rgb * visible_mask    (atlas-space)
       observed_target_mask = visible_mask                 (atlas-space)
   This is exact synthetic supervision from Cranamp -- where the rendered
   render exposed atlas pixels, we know the RGB value (because the renderer
   copied the BMP value there). The observed head teaches the encoder to
   recover screen->atlas correspondence. At inference, only the main
   `prediction` head is used; the observed head is purely a training
   signal.

NOTE: V3.2 is **not** Observer->Completer chaining. The final RGB head and
the observed head share the encoder + decoder body but split via separate
1x1 output convs. The final head does **not** consume `observed_logits` as
input -- it only benefits from the encoder having been trained against the
auxiliary correspondence supervision. A true Observer->Completer would feed
the observed head's prediction into the final head as conditioning so the
final head only fills in hidden atlas pixels; that's a separate, not-yet-
implemented architecture.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .slotnet_v31 import (
    SlotNetV31,
    StyledUpBlock,
    color_transfer_default,
    conv_block,
    build_layout_maps,
)


class SlotNetV32(SlotNetV31):
    def __init__(self, atlas_profile, default_atlas, base_channels: int = 16, slot_emb_dim: int = 16):
        super().__init__(atlas_profile=atlas_profile, default_atlas=default_atlas, base_channels=base_channels, slot_emb_dim=slot_emb_dim)
        # Residual-enabled mask: 1 on slots whose loss_weight > 0, 0 elsewhere.
        H, W = atlas_profile.canvas_h, atlas_profile.canvas_w
        rem = torch.zeros((1, H, W), dtype=torch.float32)
        for slot in atlas_profile.slots:
            if float(slot.loss_weight) > 0.0:
                rem[:, slot.y:slot.y + slot.h, slot.x:slot.x + slot.w] = 1.0
        self.register_buffer("residual_enabled_mask", rem)
        # Observed-atlas head: same input channels as out_head (the final decoder
        # output is 16-dim at full resolution -- see V3.1.up5).
        last_in = self.out_head.in_channels
        self.observed_head = nn.Conv2d(last_in, 4, 1)  # 3 RGB + 1 mask logit

    def decode_residual(self, f5: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """Override: keep the decoder body, but also stash the last-stage
        feature so we can run the observed head on it without recomputing."""
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
        self._last_decoder_features = x  # cached for the observed head
        decoded = self.out_head(x)
        if decoded.shape[-2:] != (self.atlas_h, self.atlas_w):
            decoded = F.interpolate(decoded, size=(self.atlas_h, self.atlas_w), mode="bilinear", align_corners=False)
        return decoded

    def forward(self, view: torch.Tensor, residual_alpha: float = 1.0) -> dict[str, torch.Tensor]:
        f5, style = self.encode(view)
        residual_logits = self.decode_residual(f5, style)
        # Observed head on the cached last-stage features.
        observed_logits = self.observed_head(self._last_decoder_features)
        if observed_logits.shape[-2:] != (self.atlas_h, self.atlas_w):
            observed_logits = F.interpolate(observed_logits, size=(self.atlas_h, self.atlas_w), mode="bilinear", align_corners=False)

        prior_rgb = color_transfer_default(self.default_atlas, view)
        prior_logit = torch.logit(prior_rgb.clamp(0.005, 0.995))
        # Residual gating: zero residual outside V0-trainable slots so the
        # untrained regions stay exactly at the prior.
        gated_residual = residual_alpha * residual_logits[:, :3] * self.residual_enabled_mask
        rgb_logits = prior_logit + gated_residual
        special_logits = residual_logits[:, 3:]
        prediction = torch.cat((rgb_logits, special_logits), dim=1)
        return {
            "prediction": prediction,
            "prior_rgb": prior_rgb,
            "residual_logits": residual_logits,
            "observed_logits": observed_logits,  # [B, 4, H, W]: 3 RGB logits + 1 mask logit
        }


def observed_atlas_loss(
    observed_logits: torch.Tensor,    # [B, 4, H, W]
    target_rgb: torch.Tensor,          # [B, 3, H, W] in [0, 1]
    visible_mask: torch.Tensor,        # [B, 1, H, W] in {0, 1}
    atlas_mask: torch.Tensor,          # [B, 1, H, W] in {0, 1}
) -> dict[str, torch.Tensor]:
    """Auxiliary supervision teaching screen->atlas correspondence.

    observed_target_rgb = target_rgb where visible AND inside atlas_mask
    observed_target_mask = visible_mask * atlas_mask

    Returns dict with `total`, `rgb_l1`, `mask_bce`.
    """
    pred_rgb = observed_logits[:, :3].sigmoid()
    pred_mask = observed_logits[:, 3:4]  # raw logits for BCE
    valid = atlas_mask  # only score inside the atlas slot region
    eval_rgb_mask = valid * visible_mask
    denom = eval_rgb_mask.sum().clamp_min(1e-8)
    l_rgb = ((pred_rgb - target_rgb).abs() * eval_rgb_mask).sum() / (denom * pred_rgb.shape[1])
    # Mask BCE: target = visible_mask (inside atlas only contributes a learning
    # signal -- outside-atlas pixels are ignored by valid weighting).
    target_mask = visible_mask
    bce_full = F.binary_cross_entropy_with_logits(pred_mask, target_mask, reduction="none")
    denom2 = valid.sum().clamp_min(1e-8)
    l_mask = (bce_full * valid).sum() / denom2
    total = l_rgb + l_mask
    return {"total": total, "rgb_l1": l_rgb, "mask_bce": l_mask}


__all__ = ["SlotNetV32", "observed_atlas_loss"]
