from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import zipfile

from PIL import Image


@dataclass(frozen=True)
class SkinAsset:
    canonical_name: str
    original_path: str
    data: bytes


def normalize_name(name: str) -> str:
    return name.replace("\\", "/").rsplit("/", 1)[-1].strip().lower()


def canonical_display_name(name: str) -> str:
    normalized = normalize_name(name)
    if normalized.endswith(".bmp"):
        return normalized[:-4].upper() + ".bmp"
    if normalized.endswith(".txt"):
        return normalized[:-4].upper() + ".TXT"
    return normalized


def stable_skin_id(source_path: str | Path) -> str:
    path = Path(source_path)
    base = re.sub(r"[^a-zA-Z0-9]+", "_", path.stem or path.name).strip("_").lower()
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"{base or 'skin'}_{digest}"


def discover_skin_sources(skins_raw: str | Path) -> list[Path]:
    root = Path(skins_raw)
    if not root.exists():
        return []
    children = sorted(root.iterdir(), key=lambda p: p.name.lower())
    sources = [p for p in children if p.is_dir() or p.suffix.lower() == ".wsz"]
    if not sources and _looks_like_skin_dir(root):
        return [root]
    return sources


def load_skin_assets(source_path: str | Path) -> dict[str, SkinAsset]:
    source = Path(source_path)
    if source.is_dir():
        return _load_dir_assets(source)
    if source.suffix.lower() == ".wsz":
        return _load_wsz_assets(source)
    raise ValueError(f"unsupported skin source: {source}")


def load_rgb_image(asset: SkinAsset) -> Image.Image:
    from io import BytesIO

    with Image.open(BytesIO(asset.data)) as img:
        return img.convert("RGB")


def load_default_assets(default_skin: str | Path) -> dict[str, SkinAsset]:
    return _load_dir_assets(Path(default_skin))


def _looks_like_skin_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(normalize_name(child.name) == "main.bmp" for child in path.iterdir() if child.is_file())


def _load_dir_assets(path: Path) -> dict[str, SkinAsset]:
    assets: dict[str, SkinAsset] = {}
    for file_path in sorted(path.rglob("*"), key=lambda p: str(p).lower()):
        if not file_path.is_file():
            continue
        key = normalize_name(file_path.name)
        if key in assets:
            continue
        assets[key] = SkinAsset(
            canonical_name=canonical_display_name(file_path.name),
            original_path=str(file_path),
            data=file_path.read_bytes(),
        )
    return assets


def _load_wsz_assets(path: Path) -> dict[str, SkinAsset]:
    assets: dict[str, SkinAsset] = {}
    with zipfile.ZipFile(path) as archive:
        for info in sorted(archive.infolist(), key=lambda i: i.filename.lower()):
            if info.is_dir():
                continue
            key = normalize_name(info.filename)
            if key in assets:
                continue
            assets[key] = SkinAsset(
                canonical_name=canonical_display_name(info.filename),
                original_path=f"{path}!{info.filename}",
                data=archive.read(info),
            )
    return assets

