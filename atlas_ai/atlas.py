from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .profiles import AtlasProfile
from .skins import SkinAsset, load_rgb_image, normalize_name, stable_skin_id


ZERO_LOSS_FILES = {"numbers.bmp", "text.bmp", "eq_ex.bmp", "gen.bmp", "video.bmp"}


@dataclass
class PackedSkin:
    skin_id: str
    atlas: Image.Image
    mask: Image.Image
    slot_weights: np.ndarray
    metadata: dict
    rejected_reason: str | None = None


def pack_skin_assets(
    source_path: str | Path,
    assets: dict[str, SkinAsset],
    default_assets: dict[str, SkinAsset],
    atlas_profile: AtlasProfile,
) -> PackedSkin:
    skin_id = stable_skin_id(source_path)
    atlas = Image.new("RGB", (atlas_profile.canvas_w, atlas_profile.canvas_h), (0, 0, 0))
    mask = Image.new("L", (atlas_profile.canvas_w, atlas_profile.canvas_h), 0)
    weights = np.zeros((len(atlas_profile.slots),), dtype="<f4")
    metadata = {
        "skin_id": skin_id,
        "source_path": str(source_path),
        "slots": {},
    }

    if "main.bmp" not in assets:
        return PackedSkin(
            skin_id=skin_id,
            atlas=atlas,
            mask=mask,
            slot_weights=weights,
            metadata=metadata,
            rejected_reason="missing MAIN.bmp",
        )

    for slot in atlas_profile.slots:
        if slot.file is None:
            metadata["slots"][slot.name] = {"status": "reserved", "weight_multiplier": 0.0}
            continue

        key = normalize_name(slot.file)
        source_asset = assets.get(key)
        default_asset = default_assets.get(key)
        asset = source_asset or default_asset
        status = "source" if source_asset else "default_missing"
        multiplier = 1.0 if source_asset else 0.25

        if key in ZERO_LOSS_FILES or slot.loss_weight == 0.0:
            multiplier = 0.0

        if asset is None:
            metadata["slots"][slot.name] = {
                "file": slot.file,
                "status": "missing_no_default",
                "weight_multiplier": 0.0,
            }
            continue

        image = load_rgb_image(asset)
        if image.width > slot.w or image.height > slot.h:
            metadata["slots"][slot.name] = {
                "file": slot.file,
                "status": "oversize",
                "source_path": asset.original_path,
                "size": [image.width, image.height],
                "capacity": [slot.w, slot.h],
                "weight_multiplier": 0.0,
            }
            continue

        atlas.paste(image, (slot.x, slot.y))
        slot_mask = Image.new("L", image.size, 255)
        mask.paste(slot_mask, (slot.x, slot.y))
        weights[slot.id] = multiplier
        metadata["slots"][slot.name] = {
            "file": slot.file,
            "status": status,
            "source_path": asset.original_path,
            "size": [image.width, image.height],
            "capacity": [slot.w, slot.h],
            "atlas_rect": [slot.x, slot.y, slot.x + slot.w, slot.y + slot.h],
            "pasted_rect": [slot.x, slot.y, slot.x + image.width, slot.y + image.height],
            "weight_multiplier": float(multiplier),
        }

    return PackedSkin(
        skin_id=skin_id,
        atlas=atlas,
        mask=mask,
        slot_weights=weights,
        metadata=metadata,
    )


def save_packed_skin(packed: PackedSkin, out_dir: str | Path) -> dict[str, str]:
    from .profiles import write_json

    out = Path(out_dir)
    atlas_dir = out / "atlases"
    atlas_dir.mkdir(parents=True, exist_ok=True)

    atlas_path = atlas_dir / f"{packed.skin_id}.png"
    mask_path = atlas_dir / f"{packed.skin_id}.mask.png"
    weights_path = atlas_dir / f"{packed.skin_id}.slot_weight.f32"
    meta_path = atlas_dir / f"{packed.skin_id}.meta.json"

    packed.atlas.save(atlas_path)
    packed.mask.save(mask_path)
    packed.slot_weights.astype("<f4").tofile(weights_path)
    write_json(meta_path, packed.metadata)

    return {
        "atlas_path": str(atlas_path),
        "mask_path": str(mask_path),
        "slot_weight_path": str(weights_path),
        "meta_path": str(meta_path),
    }

