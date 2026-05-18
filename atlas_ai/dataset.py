from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SampleRow:
    skin_id: str
    variant_id: str
    view_png: str
    atlas_png: str
    meta_json: str


def read_sample_csv(path: str | Path) -> list[SampleRow]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        rows = []
        for row in csv.DictReader(f):
            rows.append(SampleRow(**{field: row[field] for field in SampleRow.__dataclass_fields__}))
    return rows


def image_to_tensor(path: str | Path, mode: str = "RGB") -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert(mode)
        array = np.asarray(image, dtype=np.float32)
    if mode == "L":
        return torch.from_numpy(array[None, :, :] / 255.0)
    return torch.from_numpy(array.transpose(2, 0, 1) / 255.0)


class RenderDataset(Dataset):
    def __init__(self, csv_path: str | Path):
        self.rows = read_sample_csv(csv_path)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[index]
        return {
            "skin_id": row.skin_id,
            "variant_id": row.variant_id,
            "view": image_to_tensor(row.view_png, "RGB"),
            "atlas": image_to_tensor(row.atlas_png, "RGB"),
        }
