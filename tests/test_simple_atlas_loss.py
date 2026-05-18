"""Autograd tests for the V3.4 image-to-atlas loss and dataset path."""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from models.losses import simple_atlas_loss
from train_slotnet import AtlasPairDataset, collate_atlas_pairs


def test_simple_atlas_loss_is_plain_rgb_l1():
    prediction = torch.zeros(1, 3, 2, 2, requires_grad=True)
    target_rgb = torch.ones(1, 3, 2, 2)
    losses = simple_atlas_loss(prediction, target_rgb)
    expected = (prediction.sigmoid() - target_rgb).abs().mean()
    assert torch.equal(losses["total"], expected)
    assert torch.equal(losses["rgb"], expected)
    losses["total"].backward()
    assert prediction.grad is not None
    assert torch.count_nonzero(prediction.grad).item() == prediction.numel()


def test_v34_training_data_path_uses_only_view_and_target_atlas_png(tmp_path: Path):
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
