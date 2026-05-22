"""V6CopyDataset structural tests against the generated default-skin dataset."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/16_make_v6_dataset.py"


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("v6_data")
    subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--skin-source", str(ROOT / "assets/default_skin"),
            "--skin-id", "default_skin_test",
            "--variants", "2",
            "--canvas-w", "240",
            "--canvas-h", "432",
            "--out", str(out),
        ],
        check=True,
    )
    return out


def test_dataset_loads_view_and_files(tiny_dataset):
    from atlas_ai.dataset_v6 import V6CopyDataset
    from atlas_ai.export_spec import TRAINABLE_EXPORT_SPECS

    ds = V6CopyDataset(tiny_dataset / "train.csv", ROOT / "assets/default_skin")
    assert len(ds) == 2
    sample = ds[0]
    assert sample["skin_id"] == "default_skin_test"
    assert sample["view"].shape == (3, 432, 240)
    assert sample["view"].dtype == torch.float32
    for spec in TRAINABLE_EXPORT_SPECS:
        entry = sample["files"][spec.file_name]
        assert entry["target"].shape == (3, spec.h, spec.w)
        assert entry["visible"].shape == (1, spec.h, spec.w)
        assert entry["uv"].shape == (2, spec.h, spec.w)


def test_dataset_dataloader_collate(tiny_dataset):
    """Default collate must stack all tensors and pass strings through."""
    from atlas_ai.dataset_v6 import V6CopyDataset
    from atlas_ai.export_spec import TRAINABLE_EXPORT_SPECS

    ds = V6CopyDataset(tiny_dataset / "train.csv", ROOT / "assets/default_skin")
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    assert batch["view"].shape == (2, 3, 432, 240)
    assert len(batch["skin_id"]) == 2
    for spec in TRAINABLE_EXPORT_SPECS:
        assert batch["files"][spec.file_name]["target"].shape == (2, 3, spec.h, spec.w)
        assert batch["files"][spec.file_name]["visible"].shape == (2, 1, spec.h, spec.w)
        assert batch["files"][spec.file_name]["uv"].shape == (2, 2, spec.h, spec.w)


def test_dataset_resolves_relative_csv_paths(tiny_dataset, tmp_path):
    """CSV paths may be absolute or relative to the CSV directory."""
    from atlas_ai.dataset_v6 import V6CopyDataset

    src_csv = tiny_dataset / "train.csv"
    rel_csv = tmp_path / "relative.csv"
    rows = list(csv.DictReader(src_csv.open("r", newline="", encoding="utf-8")))
    for row in rows:
        row["view_png"] = str(Path(row["view_png"]).relative_to(tiny_dataset))
        row["labels_npz"] = str(Path(row["labels_npz"]).relative_to(tiny_dataset))
    with rel_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["skin_id", "variant_id", "view_png", "labels_npz"])
        writer.writeheader()
        writer.writerows(rows)
    # Place the CSV beside the relative views/labels tree.
    colocated = tiny_dataset / "relative.csv"
    colocated.write_text(rel_csv.read_text(encoding="utf-8"), encoding="utf-8")

    ds = V6CopyDataset(colocated, ROOT / "assets/default_skin")
    sample = ds[0]
    assert sample["view"].shape == (3, 432, 240)


def test_dataset_multi_skin_csv_rejected(tiny_dataset, tmp_path):
    """One-skin-only constraint: a CSV with two skin_ids must raise."""
    from atlas_ai.dataset_v6 import V6CopyDataset

    src_csv = tiny_dataset / "train.csv"
    bad_csv = tmp_path / "two_skin.csv"
    text = src_csv.read_text()
    # Append a row with a different skin_id; reuse same view/labels paths.
    lines = text.splitlines()
    sample_row = lines[1].split(",")
    sample_row[0] = "another_skin"
    bad_csv.write_text("\n".join(lines + [",".join(sample_row)]) + "\n")
    with pytest.raises(ValueError, match="one-skin only"):
        V6CopyDataset(bad_csv, ROOT / "assets/default_skin")
