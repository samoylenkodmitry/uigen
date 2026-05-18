from __future__ import annotations

from pathlib import Path

from PIL import Image

from .profiles import AtlasProfile
from .skins import SkinAsset, load_rgb_image, normalize_name, stable_skin_id


class PackedSkin:
    def __init__(
        self,
        skin_id: str,
        atlas: Image.Image,
        metadata: dict,
        rejected_reason: str | None = None,
    ):
        self.skin_id = skin_id
        self.atlas = atlas
        self.metadata = metadata
        self.rejected_reason = rejected_reason


def pack_skin_assets(
    source_path: str | Path,
    assets: dict[str, SkinAsset],
    default_assets: dict[str, SkinAsset],
    atlas_profile: AtlasProfile,
) -> PackedSkin:
    skin_id = stable_skin_id(source_path)
    atlas = Image.new("RGB", (atlas_profile.canvas_w, atlas_profile.canvas_h), (0, 0, 0))
    metadata = {
        "skin_id": skin_id,
        "source_path": str(source_path),
        "slots": {},
    }

    if "main.bmp" not in assets:
        return PackedSkin(
            skin_id=skin_id,
            atlas=atlas,
            metadata=metadata,
            rejected_reason="missing MAIN.bmp",
        )

    for slot in atlas_profile.slots:
        if slot.file is None:
            metadata["slots"][slot.name] = {"status": "reserved"}
            continue

        key = normalize_name(slot.file)
        source_asset = assets.get(key)
        default_asset = default_assets.get(key)
        asset = source_asset or default_asset
        status = "source" if source_asset else "default_missing"

        if asset is None:
            metadata["slots"][slot.name] = {
                "file": slot.file,
                "status": "missing_no_default",
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
            }
            continue

        atlas.paste(image, (slot.x, slot.y))
        metadata["slots"][slot.name] = {
            "file": slot.file,
            "status": status,
            "source_path": asset.original_path,
            "size": [image.width, image.height],
            "capacity": [slot.w, slot.h],
            "atlas_rect": [slot.x, slot.y, slot.x + slot.w, slot.y + slot.h],
            "pasted_rect": [slot.x, slot.y, slot.x + image.width, slot.y + image.height],
        }

    return PackedSkin(
        skin_id=skin_id,
        atlas=atlas,
        metadata=metadata,
    )


def save_packed_skin(packed: PackedSkin, out_dir: str | Path) -> dict[str, str]:
    from .profiles import write_json

    out = Path(out_dir)
    atlas_dir = out / "atlases"
    atlas_dir.mkdir(parents=True, exist_ok=True)

    atlas_path = atlas_dir / f"{packed.skin_id}.png"
    meta_path = atlas_dir / f"{packed.skin_id}.meta.json"

    packed.atlas.save(atlas_path)
    write_json(meta_path, packed.metadata)

    return {
        "atlas_path": str(atlas_path),
        "meta_path": str(meta_path),
    }
