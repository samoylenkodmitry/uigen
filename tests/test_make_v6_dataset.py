"""End-to-end test for scripts/16_make_v6_dataset.py."""

from __future__ import annotations

import csv
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/16_make_v6_dataset.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("make_v6_dataset", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_stable_seed_matches_render_dataset_script():
    # Match scripts/02_render_dataset.py so V6 datasets line up with V35 splits.
    spec = importlib.util.spec_from_file_location(
        "render_dataset", ROOT / "scripts/02_render_dataset.py"
    )
    rd = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(rd)
    mod = _load_script_module()
    for skin_id in ["skin_180ffb08", "darkside_127876f0", "abc_def"]:
        for variant_id in [0, 1, 31, 1023]:
            assert rd.stable_seed(skin_id, variant_id) == mod.stable_seed(skin_id, variant_id)


def test_dataset_generation_end_to_end(tmp_path):
    out = tmp_path / "v6_data"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--skin-source", str(ROOT / "assets/default_skin"),
            "--skin-id", "default_skin_test",
            "--variants", "2",
            "--canvas-w", "240",
            "--canvas-h", "432",
            "--out", str(out),
        ],
        check=True,
    )
    csv_path = out / "train.csv"
    assert csv_path.exists()
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    for row in rows:
        assert row["skin_id"] == "default_skin_test"
        assert row["variant_id"] in {"0000", "0001"}
        view_path = Path(row["view_png"])
        labels_path = Path(row["labels_npz"])
        assert view_path.is_absolute()
        assert labels_path.is_absolute()
        assert view_path.exists()
        assert labels_path.exists()
        # View shape matches canvas.
        with Image.open(view_path) as im:
            assert im.size == (240, 432)
            assert im.mode == "RGB"


def test_labels_match_view_via_oracle_copy(tmp_path):
    """Generated labels must reproduce clean BMP pixels via the V6 copy path."""
    from atlas_ai.v6_labels import load_v6_labels
    from atlas_ai.export_spec import TRAINABLE_EXPORT_SPECS

    out = tmp_path / "v6_data"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--skin-source", str(ROOT / "assets/default_skin"),
            "--skin-id", "default_skin_test",
            "--variants", "1",
            "--canvas-w", "240",
            "--canvas-h", "432",
            "--out", str(out),
        ],
        check=True,
    )
    view_path = out / "views/default_skin_test_0000.png"
    labels_path = out / "labels/default_skin_test_0000.npz"
    with Image.open(view_path) as im:
        canvas_rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
    labels = load_v6_labels(labels_path)

    spec_by_name = {spec.file_name: spec for spec in TRAINABLE_EXPORT_SPECS}
    default_skin = ROOT / "assets/default_skin"

    total_visible = 0
    total_match = 0
    for file_name, entry in labels.items():
        visible = entry["visible_mask"]
        if visible.sum() == 0:
            continue
        spec = spec_by_name[file_name]
        with Image.open(default_skin / file_name) as im:
            clean = np.asarray(im.convert("RGB"), dtype=np.uint8)
        assert clean.shape[:2] == (spec.h, spec.w)
        view = torch.from_numpy(canvas_rgb.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        uv = torch.from_numpy(entry["uv_target"].astype(np.float32)).unsqueeze(0).permute(0, 2, 3, 1)
        vmask3 = np.repeat(visible[:, :, None], 3, axis=2).astype(bool)
        for mode in ("nearest", "bilinear"):
            sampled = F.grid_sample(view, uv, mode=mode, align_corners=False).squeeze(0)
            sampled_rgb = (sampled.clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
            delta = np.abs(sampled_rgb.astype(np.int16) - clean.astype(np.int16))
            match = (delta[vmask3] <= 1).sum()
            total_visible += int(vmask3.sum())
            total_match += int(match)
    assert total_visible > 0
    assert total_match / total_visible > 0.995, (
        f"label oracle match: {total_match}/{total_visible} "
        f"({total_match/total_visible*100:.2f}%)"
    )


def test_dataset_generation_is_deterministic(tmp_path):
    """Two runs with the same seed produce bit-identical views and labels."""
    runs = []
    for run in ("a", "b"):
        out = tmp_path / run
        subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--skin-source", str(ROOT / "assets/default_skin"),
                "--skin-id", "default_skin_test",
                "--variants", "1",
                "--canvas-w", "240",
                "--canvas-h", "432",
                "--out", str(out),
            ],
            check=True,
        )
        runs.append(out)
    view_a = (runs[0] / "views/default_skin_test_0000.png").read_bytes()
    view_b = (runs[1] / "views/default_skin_test_0000.png").read_bytes()
    assert view_a == view_b
    labels_a = np.load(runs[0] / "labels/default_skin_test_0000.npz")
    labels_b = np.load(runs[1] / "labels/default_skin_test_0000.npz")
    assert sorted(labels_a.files) == sorted(labels_b.files)
    for key in labels_a.files:
        assert np.array_equal(labels_a[key], labels_b[key]), key
