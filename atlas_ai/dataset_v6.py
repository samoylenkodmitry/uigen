"""V6 source-preserving training dataset.

Each sample provides:

    view:        [3, H_view, W_view] in [0, 1]
    files:       dict keyed by exported BMP name. Each entry:
        target:  [3, H_f, W_f]      clean BMP from the skin source, in [0, 1]
        visible: [1, H_f, W_f]      0/1 mask of pixels with a known UV target
        uv:      [2, H_f, W_f]      ground-truth UV in [-1, 1], align_corners=False
                                    convention (channel 0 = u/x, 1 = v/y)

For Stage 2 (one skin), the same clean BMP set is shared across all variants and
loaded once in __init__. Multi-skin support (Stage 3+) requires extending the
constructor with a skin_id -> skin_source mapping; that path is intentionally
deferred to keep Stage 2 minimal.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from atlas_ai.export_spec import TRAINABLE_EXPORT_SPECS
from atlas_ai.v6_labels import load_v6_labels


def _read_v6_rows(csv_path: Path) -> list[dict[str, str]]:
    with Path(csv_path).open("r", newline="", encoding="utf-8") as f:
        return [row for row in csv.DictReader(f)]


def _resolve_csv_path(csv_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else csv_dir / path


def _load_clean_targets(skin_source: Path) -> dict[str, torch.Tensor]:
    """Load every trainable exported BMP from a skin source directory.

    Returns: dict[file_name -> Tensor[3, H, W] in [0, 1]].
    """
    skin_source = Path(skin_source)
    targets: dict[str, torch.Tensor] = {}
    for spec in TRAINABLE_EXPORT_SPECS:
        path = skin_source / spec.file_name
        if not path.exists():
            raise FileNotFoundError(f"missing trainable BMP for skin source {skin_source}: {spec.file_name}")
        with Image.open(path) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        if arr.shape[:2] != (spec.h, spec.w):
            raise ValueError(
                f"{path} shape {arr.shape[:2]} != spec ({spec.h}, {spec.w})"
            )
        targets[spec.file_name] = torch.from_numpy(arr.transpose(2, 0, 1).copy()).contiguous()
    return targets


class V6CopyDataset(Dataset):
    """One-skin V6 Stage 2 dataset.

    Args:
        csv_path: train.csv emitted by scripts/16_make_v6_dataset.py.
        skin_source: directory containing the clean exported BMPs for this skin.
            All rows in csv_path must reference the same skin.
    """

    def __init__(self, csv_path: str | Path, skin_source: str | Path):
        self.csv_path = Path(csv_path)
        self.csv_dir = self.csv_path.parent
        self.rows = _read_v6_rows(self.csv_path)
        if not self.rows:
            raise ValueError(f"{csv_path} contained no rows")
        skin_ids = {row["skin_id"] for row in self.rows}
        if len(skin_ids) != 1:
            raise ValueError(
                f"V6CopyDataset is one-skin only; CSV has skin ids: {sorted(skin_ids)}"
            )
        self.skin_id = next(iter(skin_ids))
        self.targets = _load_clean_targets(Path(skin_source))

    def __len__(self) -> int:
        return len(self.rows)

    def _load_view(self, path: str | Path) -> torch.Tensor:
        with Image.open(path) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(arr.transpose(2, 0, 1).copy()).contiguous()

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        view_path = _resolve_csv_path(self.csv_dir, row["view_png"])
        labels_path = _resolve_csv_path(self.csv_dir, row["labels_npz"])
        view = self._load_view(view_path)
        labels = load_v6_labels(labels_path)
        files: dict[str, dict[str, torch.Tensor]] = {}
        for spec in TRAINABLE_EXPORT_SPECS:
            entry = labels.get(spec.file_name)
            if entry is None:
                raise KeyError(
                    f"labels file {labels_path} missing entry for {spec.file_name}"
                )
            visible = torch.from_numpy(entry["visible_mask"].astype(np.float32)).unsqueeze(0)
            uv = torch.from_numpy(entry["uv_target"].astype(np.float32))
            files[spec.file_name] = {
                "target": self.targets[spec.file_name],
                "visible": visible,
                "uv": uv,
            }
        return {
            "skin_id": row["skin_id"],
            "variant_id": row["variant_id"],
            "view": view,
            "files": files,
        }


__all__ = ["V6CopyDataset"]
