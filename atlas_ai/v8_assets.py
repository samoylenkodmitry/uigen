"""V8 tensor asset IO and skin packaging helpers."""

from __future__ import annotations

from pathlib import Path
import shutil
import zipfile

import numpy as np
import torch
from PIL import Image

from atlas_ai.export_spec import TRAINABLE_EXPORT_SPECS
from atlas_ai.skins import canonical_display_name, normalize_name


RUNTIME_FALLBACK_FILES = {
    "NUMBERS.bmp",
    "TEXT.bmp",
    "PLEDIT.TXT",
    "VISCOLOR.TXT",
}


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1).copy()).contiguous()


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    if tensor.dim() != 3 or tensor.shape[0] != 3:
        raise ValueError(f"expected [3,H,W] tensor, got {tuple(tensor.shape)}")
    arr = tensor.detach().clamp(0.0, 1.0).cpu().numpy()
    arr = (arr.transpose(1, 2, 0) * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def load_exported_tensors(
    skin_dir: str | Path,
    *,
    default_skin: str | Path = "assets/default_skin",
) -> dict[str, torch.Tensor]:
    """Load trainable exported BMP tensors, falling back to the default skin."""
    skin_dir = Path(skin_dir)
    default_skin = Path(default_skin)
    out: dict[str, torch.Tensor] = {}
    for spec in TRAINABLE_EXPORT_SPECS:
        path = skin_dir / spec.file_name
        if not path.exists():
            path = default_skin / spec.file_name
        with Image.open(path) as image:
            if image.size != (spec.w, spec.h):
                raise ValueError(f"{path} size {image.size} != {(spec.w, spec.h)}")
            out[spec.file_name] = image_to_tensor(image)
    return out


def save_exported_tensors(
    files: dict[str, torch.Tensor],
    out_dir: str | Path,
    *,
    default_skin: str | Path = "assets/default_skin",
    package: bool = True,
) -> Path | None:
    """Write trainable BMPs plus runtime defaults. Returns skin.wsz when packaged."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    default_skin = Path(default_skin)
    written: set[str] = set()
    for spec in TRAINABLE_EXPORT_SPECS:
        tensor = files.get(spec.file_name)
        if tensor is None:
            with Image.open(default_skin / spec.file_name) as image:
                tensor = image_to_tensor(image)
        if tensor.shape != (3, spec.h, spec.w):
            raise ValueError(f"{spec.file_name} tensor shape {tuple(tensor.shape)} != {(3, spec.h, spec.w)}")
        out_name = canonical_display_name(spec.file_name)
        tensor_to_image(tensor).save(out / out_name, format="BMP")
        written.add(normalize_name(out_name))

    for source in sorted(default_skin.iterdir(), key=lambda p: p.name.lower()):
        if not source.is_file():
            continue
        name = canonical_display_name(source.name)
        if name not in RUNTIME_FALLBACK_FILES:
            continue
        if normalize_name(name) in written:
            continue
        shutil.copy2(source, out / name)
        written.add(normalize_name(name))

    if not package:
        return None
    return package_skin_dir(out)


def package_skin_dir(skin_dir: str | Path, out_path: str | Path | None = None) -> Path:
    skin_dir = Path(skin_dir)
    zip_path = Path(out_path) if out_path else skin_dir / "skin.wsz"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    allowed_ext = {".bmp", ".txt", ".cur", ".ani"}
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(skin_dir.iterdir(), key=lambda p: p.name.lower()):
            if path == zip_path or not path.is_file() or path.suffix.lower() not in allowed_ext:
                continue
            archive.write(path, arcname=path.name)
    return zip_path


__all__ = [
    "RUNTIME_FALLBACK_FILES",
    "image_to_tensor",
    "load_exported_tensors",
    "package_skin_dir",
    "save_exported_tensors",
    "tensor_to_image",
]
