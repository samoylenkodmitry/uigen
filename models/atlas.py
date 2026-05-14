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


__all__ = ["pack_skin_assets", "crop_slot", "load_slot_target", "magenta_policy_for_slot", "load_atlas_profile", "load_json"]
