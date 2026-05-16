from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from atlas_ai.atlas import pack_skin_assets
from atlas_ai.dataset import image_to_tensor
from atlas_ai.profiles import AtlasProfile, Slot, load_atlas_profile, load_json


def crop_slot(tensor: torch.Tensor, slot: Slot) -> torch.Tensor:
    return tensor[..., slot.y : slot.y + slot.h, slot.x : slot.x + slot.w]


def magenta_policy_for_slot(policy: dict, slot_name: str) -> bool:
    return bool(policy.get("per_slot", {}).get(slot_name, policy.get("default", False)))


def load_slot_target(
    atlas_png: str | Path,
    atlas_mask_png: str | Path,
    visible_mask_png: str | Path,
    slot_weight_f32: str | Path,
    slot_name: str,
    atlas_profile: AtlasProfile,
    magenta_policy: dict | None = None,
) -> dict[str, torch.Tensor | bool | float]:
    slot = atlas_profile.slots_by_name[slot_name]
    atlas = image_to_tensor(atlas_png, "RGB").unsqueeze(0)
    atlas_mask = image_to_tensor(atlas_mask_png, "L").unsqueeze(0)
    visible_mask = image_to_tensor(visible_mask_png, "L").unsqueeze(0)
    weights = np.fromfile(slot_weight_f32, dtype="<f4")
    target_rgb = crop_slot(atlas, slot)
    mask = crop_slot(atlas_mask, slot)
    visible = crop_slot(visible_mask, slot)
    effective = mask * (0.25 + 0.75 * visible)

    special_enabled = magenta_policy_for_slot(magenta_policy or {"default": False}, slot.name)
    special = torch.zeros((1, slot.h, slot.w), dtype=torch.long)
    if special_enabled:
        magenta = (
            (target_rgb[:, 0] > 0.999)
            & (target_rgb[:, 1] < 0.001)
            & (target_rgb[:, 2] > 0.999)
            & (mask[:, 0] > 0.5)
        )
        special[magenta] = 1

    return {
        "target_rgb": target_rgb,
        "atlas_mask": mask,
        "visible_mask": visible,
        "effective_mask": effective,
        "special_target": special,
        "special_enabled": special_enabled,
        "slot_weight": float(slot.loss_weight * weights[slot.id]),
    }


def pack_default_atlas_tensor(default_skin: str | Path, atlas_profile: AtlasProfile) -> torch.Tensor:
    """Pack the default-skin BMPs into the atlas v1 layout and return [3, H, W] in [0, 1]."""
    from atlas_ai.skins import load_default_assets
    assets = load_default_assets(default_skin)
    packed = pack_skin_assets(default_skin, assets, assets, atlas_profile)
    arr = np.asarray(packed.atlas.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def load_full_atlas_target(
    atlas_png: str | Path,
    atlas_mask_png: str | Path,
    visible_mask_png: str | Path,
    slot_weight_f32: str | Path,
    atlas_profile: AtlasProfile,
    magenta_policy: dict | None = None,
    hidden_weight: float = 0.03,
    visible_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Return full-atlas targets at 1024x1024 (no slot cropping).

    Built for SlotNetV3 which predicts the entire atlas in one shot.

    Returns:
      target_rgb        [3, H, W] in [0, 1]
      atlas_mask        [1, H, W] in {0, 1}  -- where real BMP pixels exist
      visible_mask      [1, H, W] in {0, 1}  -- where the current view sampled
      effective_mask    [1, H, W] in [0, 1]  -- atlas_mask * (0.25 + 0.75*vis)
      weight_map        [1, H, W] >= 0       -- per-pixel slot loss weight
      special_target    [H, W] long           -- 1 where magenta target (only inside
                                                magenta-enabled slots), else 0
      special_mask      [1, H, W] in {0, 1}  -- where to apply special CE
    """
    H = atlas_profile.canvas_h
    W = atlas_profile.canvas_w
    atlas = image_to_tensor(atlas_png, "RGB")          # [3, H, W]
    atlas_mask = image_to_tensor(atlas_mask_png, "L")  # [1, H, W]
    visible_mask = image_to_tensor(visible_mask_png, "L")  # [1, H, W]
    weights = np.fromfile(slot_weight_f32, dtype="<f4")
    # GPT review patch: lower hidden-pixel weight so the model isn't punished
    # for failing to predict sprite states the current render did not show.
    # Defaults shift from (0.25 + 0.75*vis) -> (0.03 + 0.97*vis).
    effective = atlas_mask * (hidden_weight + (visible_weight - hidden_weight) * visible_mask)

    weight_map = torch.zeros((1, H, W), dtype=torch.float32)
    special_target = torch.zeros((H, W), dtype=torch.long)
    special_mask = torch.zeros((1, H, W), dtype=torch.float32)
    policy = magenta_policy or {"default": False}
    for slot in atlas_profile.slots:
        w_slot = float(slot.loss_weight) * float(weights[slot.id])
        if w_slot <= 0.0:
            continue
        y0, y1 = slot.y, slot.y + slot.h
        x0, x1 = slot.x, slot.x + slot.w
        weight_map[:, y0:y1, x0:x1] = w_slot
        if magenta_policy_for_slot(policy, slot.name):
            special_mask[:, y0:y1, x0:x1] = 1.0
            region = atlas[:, y0:y1, x0:x1]
            mag = (
                (region[0] > 0.999)
                & (region[1] < 0.001)
                & (region[2] > 0.999)
                & (atlas_mask[0, y0:y1, x0:x1] > 0.5)
            )
            special_target[y0:y1, x0:x1] = torch.where(
                mag, torch.tensor(1, dtype=torch.long), torch.tensor(0, dtype=torch.long)
            )

    return {
        "target_rgb": atlas,
        "atlas_mask": atlas_mask,
        "visible_mask": visible_mask,
        "effective_mask": effective,
        "weight_map": weight_map,
        "special_target": special_target,
        "special_mask": special_mask,
    }


__all__ = [
    "pack_skin_assets", "crop_slot", "load_slot_target", "load_full_atlas_target",
    "magenta_policy_for_slot", "load_atlas_profile", "load_json",
]
