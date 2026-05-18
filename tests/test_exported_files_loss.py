"""Autograd tests for the SlotNet loss and dataset path."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from atlas_ai.export_spec import ExportFileSpec, load_file_weight_overrides, with_file_weights
from models.losses import exported_files_loss
from train_slotnet import AtlasPairDataset, collate_atlas_pairs


def test_exported_files_loss_ignores_non_exported_atlas_pixels():
    spec = ExportFileSpec("MAIN.bmp", "MAIN", x=1, y=1, w=2, h=2, weight=1.0)
    prediction = {"MAIN.bmp": torch.full((1, 3, 2, 2), -4.0, requires_grad=True)}
    target_rgb = torch.zeros(1, 3, 5, 5)

    base = exported_files_loss(prediction, target_rgb, specs=(spec,), edge_weight=0.0)["total"]
    padded_change = target_rgb.clone()
    padded_change[..., 0, :] = 1.0
    padded = exported_files_loss(prediction, padded_change, specs=(spec,), edge_weight=0.0)["total"]
    crop_change = target_rgb.clone()
    crop_change[..., 1:3, 1:3] = 1.0
    cropped = exported_files_loss(prediction, crop_change, specs=(spec,), edge_weight=0.0)["total"]

    assert torch.equal(base, padded)
    assert cropped > base
    base.backward()
    assert prediction["MAIN.bmp"].grad is not None
    assert torch.count_nonzero(prediction["MAIN.bmp"].grad).item() == prediction["MAIN.bmp"].numel()


def test_training_data_path_uses_only_view_and_target_atlas_png(tmp_path: Path):
    view_path = tmp_path / "input.png"
    atlas_path = tmp_path / "expected.png"
    Image.new("RGB", (8, 8), (1, 2, 3)).save(view_path)
    Image.new("RGB", (4, 4), (10, 20, 30)).save(atlas_path)
    csv_path = tmp_path / "train.csv"
    csv_path.write_text(
        "skin_id,variant_id,view_png,atlas_png,meta_json\n"
        f"s,0000,{view_path},{atlas_path},{tmp_path / 'missing-meta.json'}\n",
        encoding="utf-8",
    )

    item = AtlasPairDataset(csv_path)[0]
    batch = collate_atlas_pairs([item])

    assert set(batch) == {"view", "target_rgb"}
    assert batch["view"].shape == (1, 3, 8, 8)
    assert batch["target_rgb"].shape == (1, 3, 4, 4)


def test_file_weight_override_yaml_accepts_stems_and_bmp_names(tmp_path: Path):
    path = tmp_path / "weights.yaml"
    path.write_text(
        "file_weights:\n"
        "  MAIN: 1.5\n"
        "  EQMAIN.bmp: 2.0\n",
        encoding="utf-8",
    )

    specs = with_file_weights(load_file_weight_overrides(path))
    weights = {spec.file_name: spec.weight for spec in specs}

    assert weights["MAIN.bmp"] == 1.5
    assert weights["EQMAIN.bmp"] == 2.0
    assert weights["CBUTTONS.bmp"] == 4.0


def test_file_weight_override_rejects_unknown_files(tmp_path: Path):
    path = tmp_path / "weights.yaml"
    path.write_text("file_weights:\n  VIDEO.bmp: 9\n", encoding="utf-8")

    with pytest.raises(ValueError, match="VIDEO.bmp|video.bmp"):
        with_file_weights(load_file_weight_overrides(path))
