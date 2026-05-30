"""V10 BMP-expert dataset loader.

Per-BMP CSV-backed dataset: each row maps a full Cranamp render (input,
[3, 1728, 960]) to one target BMP (output, [3, H, W] at the file's exact export
size). The target is unchanged across render variants of the same skin — the
model is forced to recover the source BMP from a state-randomized view.

Built for `scripts/make_v10_bmp_expert_dataset.py` output:

    data_v10/
      renders/<skin>_<vid>.png
      targets/<skin>/<FILE>.bmp
      csv/train_<FILE>.csv     # render_png,target_bmp,skin_id,variant_id,state_json
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


CANVAS_W, CANVAS_H = 960, 1728


def _image_to_tensor(path: Path, size: tuple[int, int] | None = None) -> torch.Tensor:
    """[H,W,3] PIL -> [3,H,W] float in [0,1]; optionally resize to (W,H)."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        if size is not None and im.size != size:
            im = im.resize(size, Image.Resampling.LANCZOS)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1).copy()).contiguous()


class BMPExpertDataset(Dataset):
    """One row per (render, target BMP) pair, for a single output BMP file.

    Args:
        root: dataset root (the --out passed to the generator).
        bmp_file_name: trainable BMP filename (e.g., "MAIN.bmp").
        csv_path: optional explicit CSV; default `root / csv / train_<stem>.csv`.
    """

    def __init__(self, root: str | Path, bmp_file_name: str, *,
                 csv_path: str | Path | None = None):
        self.root = Path(root)
        self.bmp_file_name = bmp_file_name
        csv_p = Path(csv_path) if csv_path else self.root / "csv" / f"train_{Path(bmp_file_name).stem}.csv"
        if not csv_p.exists():
            raise FileNotFoundError(f"V10 CSV not found: {csv_p}")
        self.rows: list[dict] = []
        with csv_p.open() as f:
            for r in csv.DictReader(f):
                if Path(r["target_bmp"]).name != bmp_file_name:
                    continue
                self.rows.append(r)
        if not self.rows:
            raise ValueError(f"no rows for {bmp_file_name} in {csv_p}")
        # Resolve target size once from the first row (every row of a per-BMP
        # CSV writes to the same target file size).
        with Image.open(self.root / self.rows[0]["target_bmp"]) as im:
            self.target_size = im.size  # (W, H)
        # Use the render's NATIVE size (renders in one dataset share a canvas).
        # The old 960x1728 canvas was the ~275px skin upscaled ~3.3x — same info,
        # 10x the compute. Native-res renders (smaller canvas) train far faster
        # with no information loss. Falls back to the legacy constant if needed.
        with Image.open(self.root / self.rows[0]["render_png"]) as im:
            self.input_size = im.size  # (W, H)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        r = self.rows[index]
        render = _image_to_tensor(self.root / r["render_png"], size=self.input_size)
        target = _image_to_tensor(self.root / r["target_bmp"], size=self.target_size)
        return {
            "render": render,   # [3, 1728, 960]
            "target": target,   # [3, H, W]
            "skin_id": r["skin_id"],
            "variant_id": r["variant_id"],
            "state_json": r["state_json"],
        }


__all__ = ["BMPExpertDataset", "CANVAS_W", "CANVAS_H"]
