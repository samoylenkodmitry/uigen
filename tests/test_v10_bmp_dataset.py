"""V10 BMP dataset: generator produces a coherent set; loader returns the right
shapes for any TRAINABLE BMP."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def smoke_dataset(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("v10_ds")
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/make_v10_bmp_expert_dataset.py"),
         "--skin", str(ROOT / "assets/default_skin"),
         "--skin-id", "default", "--scale", "smoke", "--out", str(out),
         "--progress-every", "0"],
        check=True, cwd=str(ROOT),
    )
    return out


def test_dataset_layout(smoke_dataset: Path):
    from atlas_ai.export_spec import TRAINABLE_EXPORT_SPECS
    assert (smoke_dataset / "renders").is_dir()
    assert (smoke_dataset / "states").is_dir()
    assert (smoke_dataset / "targets" / "default").is_dir()
    for s in TRAINABLE_EXPORT_SPECS:
        assert (smoke_dataset / "targets" / "default" / s.file_name).exists()
        csv_p = smoke_dataset / "csv" / f"train_{Path(s.file_name).stem}.csv"
        assert csv_p.exists(), csv_p
        with csv_p.open() as f:
            rows = list(csv.DictReader(f))
        assert rows, f"no rows in {csv_p}"
        # all rows point at the same target for this skin/BMP
        assert all(Path(r["target_bmp"]).name == s.file_name for r in rows)
    assert (smoke_dataset / "renderer_gaps.json").exists()


def test_loader_returns_exact_shapes(smoke_dataset: Path):
    from atlas_ai.dataset_v10_bmp import BMPExpertDataset, CANVAS_H, CANVAS_W
    from atlas_ai.export_spec import TRAINABLE_EXPORT_SPECS
    for s in TRAINABLE_EXPORT_SPECS:
        ds = BMPExpertDataset(smoke_dataset, s.file_name)
        item = ds[0]
        assert tuple(item["render"].shape) == (3, CANVAS_H, CANVAS_W)
        assert tuple(item["target"].shape) == (3, s.h, s.w)
        assert float(item["render"].min()) >= 0.0 and float(item["render"].max()) <= 1.0
