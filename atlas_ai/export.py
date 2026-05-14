from __future__ import annotations

from pathlib import Path
import shutil
import zipfile

from PIL import Image

from .profiles import AtlasProfile
from .skins import canonical_display_name, normalize_name


def export_atlas_to_skin(
    atlas_path: str | Path,
    atlas_profile: AtlasProfile,
    export_profile: dict[str, dict[str, int | str]],
    default_skin: str | Path,
    out_dir: str | Path,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with Image.open(atlas_path) as source:
        atlas = source.convert("RGB")

    if atlas.size != (atlas_profile.canvas_w, atlas_profile.canvas_h):
        raise ValueError(
            f"atlas size {atlas.size} does not match "
            f"{atlas_profile.canvas_w}x{atlas_profile.canvas_h}"
        )

    slots = atlas_profile.slots_by_name
    written = set()
    for file_name, info in export_profile.items():
        slot_name = str(info["slot"])
        slot = slots[slot_name]
        width = int(info["w"])
        height = int(info["h"])
        if width > slot.w or height > slot.h:
            raise ValueError(f"{file_name} export size exceeds atlas slot {slot_name}")
        crop = atlas.crop((slot.x, slot.y, slot.x + width, slot.y + height))
        out_name = canonical_display_name(file_name)
        crop.save(out / out_name, format="BMP")
        written.add(normalize_name(out_name))

    default_path = Path(default_skin)
    for source in sorted(default_path.iterdir(), key=lambda p: p.name.lower()):
        if not source.is_file():
            continue
        key = normalize_name(source.name)
        if key in written:
            continue
        if source.suffix.lower() not in {".bmp", ".txt", ".cur", ".ani"}:
            continue
        shutil.copy2(source, out / source.name)
        written.add(key)

    shutil.copy2(atlas_path, out / "atlas.png")
    zip_path = out / "skin.wsz"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(out.iterdir(), key=lambda p: p.name.lower()):
            if not file_path.is_file() or file_path.name in {"skin.wsz", "atlas.png"}:
                continue
            archive.write(file_path, arcname=file_path.name)
    return zip_path

