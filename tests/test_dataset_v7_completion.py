"""V7 asset-completion dataset: shape, mask contract, determinism, errors."""

from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from atlas_ai.dataset_v7_completion import V7CompletionDataset
from atlas_ai.export_spec import TRAINABLE_EXPORT_SPECS, TRAINABLE_EXPORT_FILES
from atlas_ai.state_families import group_by_family, load_state_families
from atlas_ai.v7_masks import V7MaskWeights


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/state_families_classic.yaml"
DEFAULT_SKIN = ROOT / "assets/default_skin"


@pytest.fixture(scope="module")
def dataset():
    return V7CompletionDataset(
        skin_sources={"default": DEFAULT_SKIN},
        state_families_path=CONFIG,
    )


def test_dataset_length_and_item_keys(dataset):
    assert len(dataset) == len(TRAINABLE_EXPORT_FILES)
    sample = dataset[0]
    assert set(sample) == {
        "skin_id", "file_name",
        "target_rgb", "observed_mask", "observed_rgb",
        "mode",
    }
    assert sample["skin_id"] == "default"
    assert sample["file_name"] in TRAINABLE_EXPORT_FILES


def test_all_eleven_files_yielded(dataset):
    seen_files = {dataset[i]["file_name"] for i in range(len(dataset))}
    assert seen_files == set(TRAINABLE_EXPORT_FILES)
    # Non-trainable files must never be yielded.
    for bad_name in {"TEXT.bmp", "NUMBERS.bmp", "VIDEO.bmp", "GEN.bmp"}:
        assert bad_name not in seen_files


def test_per_file_shapes_and_dtypes(dataset):
    spec_by_name = {s.file_name: s for s in TRAINABLE_EXPORT_SPECS}
    for i in range(len(dataset)):
        item = dataset[i]
        spec = spec_by_name[item["file_name"]]
        assert item["target_rgb"].shape == (3, spec.h, spec.w)
        assert item["target_rgb"].dtype == torch.float32
        assert item["observed_mask"].shape == (1, spec.h, spec.w)
        assert item["observed_mask"].dtype == torch.float32
        assert item["observed_rgb"].shape == (3, spec.h, spec.w)
        assert item["observed_rgb"].dtype == torch.float32
        assert item["target_rgb"].min().item() >= 0.0
        assert item["target_rgb"].max().item() <= 1.0


def test_observed_rgb_equals_target_times_mask(dataset):
    for i in range(len(dataset)):
        item = dataset[i]
        expected = item["target_rgb"] * item["observed_mask"]
        assert torch.equal(item["observed_rgb"], expected), item["file_name"]


def test_mask_values_are_binary(dataset):
    for i in range(len(dataset)):
        mask = dataset[i]["observed_mask"]
        unique = torch.unique(mask)
        assert set(unique.tolist()) <= {0.0, 1.0}


def test_modes_are_valid_strings(dataset):
    valid = {"provenance", "state_family", "random_rect", "whole_file"}
    for i in range(len(dataset)):
        assert dataset[i]["mode"] in valid


def test_whole_file_mode_zeros_observed():
    """With weights forcing whole_file, observed_rgb must sum to zero."""
    weights = V7MaskWeights(provenance=0.0, state_family=0.0, random_rect=0.0, whole_file=1.0)
    ds = V7CompletionDataset(
        skin_sources={"default": DEFAULT_SKIN},
        state_families_path=CONFIG,
        mask_weights=weights,
    )
    for i in range(len(ds)):
        item = ds[i]
        assert item["mode"] == "whole_file"
        assert float(item["observed_mask"].sum()) == 0.0
        assert float(item["observed_rgb"].abs().sum()) == 0.0


def test_state_family_mode_hides_volume_siblings():
    """Force state_family; verify VOLUME slider sibling rects are hidden."""
    weights = V7MaskWeights(provenance=0.0, state_family=1.0, random_rect=0.0, whole_file=0.0)
    families = load_state_families(CONFIG)
    rects = families["VOLUME.bmp"]
    grouped = group_by_family(rects)
    frame_rects = grouped["slider_frames"]
    volume_index = TRAINABLE_EXPORT_FILES.index("VOLUME.bmp")
    revealed_seen: Counter[int] = Counter()
    for seed in range(50):
        ds = V7CompletionDataset(
            skin_sources={"default": DEFAULT_SKIN},
            state_families_path=CONFIG,
            mask_weights=weights,
            seed=seed,
        )
        item = ds[volume_index]
        assert item["file_name"] == "VOLUME.bmp"
        mask = item["observed_mask"][0].numpy()
        revealed = [
            i for i, r in enumerate(frame_rects)
            if mask[r.y : r.y + r.h, r.x : r.x + r.w].all()
        ]
        hidden = [
            i for i, r in enumerate(frame_rects)
            if (mask[r.y : r.y + r.h, r.x : r.x + r.w] == 0).all()
        ]
        # Either slider_frames family was picked (1 revealed, 27 hidden) or
        # the thumb family was picked (all 28 frames still observed).
        if len(revealed) == 28:
            continue  # thumb family this draw
        assert len(revealed) == 1
        assert len(hidden) == 27
        revealed_seen[revealed[0]] += 1
    assert revealed_seen, "state_family mode never revealed a slider frame"


def test_state_family_mode_hides_eqmain_siblings_across_rows():
    """EQMAIN has 28 slider frames in two rows but one family; hides must
    span both rows."""
    weights = V7MaskWeights(provenance=0.0, state_family=1.0, random_rect=0.0, whole_file=0.0)
    families = load_state_families(CONFIG)
    rects = families["EQMAIN.bmp"]
    grouped = group_by_family(rects)
    frames = grouped["slider_frames"]
    assert len(frames) == 28
    eqmain_index = TRAINABLE_EXPORT_FILES.index("EQMAIN.bmp")
    seen_hidden_both_rows = False
    for seed in range(200):
        ds = V7CompletionDataset(
            skin_sources={"default": DEFAULT_SKIN},
            state_families_path=CONFIG,
            mask_weights=weights,
            seed=seed,
        )
        item = ds[eqmain_index]
        mask = item["observed_mask"][0].numpy()
        revealed = [
            i for i, r in enumerate(frames)
            if mask[r.y : r.y + r.h, r.x : r.x + r.w].all()
        ]
        hidden = [
            i for i, r in enumerate(frames)
            if (mask[r.y : r.y + r.h, r.x : r.x + r.w] == 0).all()
        ]
        if len(revealed) == 1 and len(hidden) == 27:
            rows = {0 if i < 14 else 1 for i in hidden}
            if rows == {0, 1}:
                seen_hidden_both_rows = True
                break
    assert seen_hidden_both_rows, (
        "EQMAIN state_family masks never hid frames across both rows"
    )


def test_mode_distribution_matches_weights_when_provenance_provided():
    """4000 draws should be within 4% of the V7 plan weights."""
    weights = V7MaskWeights()
    # Build a tiny provenance pool so the provenance mode is available; one
    # entry per trainable file at the right (h, w).
    spec_by_name = {s.file_name: s for s in TRAINABLE_EXPORT_SPECS}
    pools = {
        s.file_name: [np.ones((s.h, s.w), dtype=np.uint8)]
        for s in TRAINABLE_EXPORT_SPECS
    }
    ds = V7CompletionDataset(
        skin_sources={"default": DEFAULT_SKIN},
        state_families_path=CONFIG,
        provenance_pools=pools,
        mask_weights=weights,
    )
    counts: Counter[str] = Counter()
    n = 4000
    rng = np.random.default_rng(0)
    for _ in range(n):
        index = int(rng.integers(0, len(ds)))
        ds.seed = int(rng.integers(0, 2**31 - 1))
        counts[ds[index]["mode"]] += 1
    expected = {
        "provenance": weights.provenance,
        "state_family": weights.state_family,
        "random_rect": weights.random_rect,
        "whole_file": weights.whole_file,
    }
    for mode, want in expected.items():
        got = counts[mode] / n
        assert abs(got - want) < 0.04, f"{mode}: expected {want:.2f} got {got:.3f}"


def test_dataset_deterministic_with_fixed_seed():
    """Two identical datasets must yield identical items at every index."""
    ds_a = V7CompletionDataset(
        skin_sources={"default": DEFAULT_SKIN},
        state_families_path=CONFIG,
        seed=42,
    )
    ds_b = V7CompletionDataset(
        skin_sources={"default": DEFAULT_SKIN},
        state_families_path=CONFIG,
        seed=42,
    )
    for i in range(len(ds_a)):
        a = ds_a[i]
        b = ds_b[i]
        assert a["mode"] == b["mode"]
        assert torch.equal(a["observed_mask"], b["observed_mask"])
        assert torch.equal(a["observed_rgb"], b["observed_rgb"])
        assert torch.equal(a["target_rgb"], b["target_rgb"])


def test_dataset_seed_changes_alter_masks():
    """Different seeds should produce at least some different masks across items."""
    ds_a = V7CompletionDataset(
        skin_sources={"default": DEFAULT_SKIN},
        state_families_path=CONFIG,
        seed=1,
    )
    ds_b = V7CompletionDataset(
        skin_sources={"default": DEFAULT_SKIN},
        state_families_path=CONFIG,
        seed=2,
    )
    differences = 0
    for i in range(len(ds_a)):
        if not torch.equal(ds_a[i]["observed_mask"], ds_b[i]["observed_mask"]):
            differences += 1
    assert differences > 0


def test_dataloader_batch_size_one(dataset):
    """Default collate must handle one-item batches with mixed tensor shapes."""
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    assert batch["target_rgb"].dim() == 4  # [B, 3, H, W]
    assert batch["target_rgb"].shape[0] == 1
    assert isinstance(batch["skin_id"], list) and len(batch["skin_id"]) == 1
    assert isinstance(batch["mode"], list) and len(batch["mode"]) == 1


def test_dataloader_same_file_batch_collates():
    """Items at the same file index across skins must batch cleanly."""
    skin_sources = {"a": DEFAULT_SKIN, "b": DEFAULT_SKIN}
    ds = V7CompletionDataset(
        skin_sources=skin_sources,
        state_families_path=CONFIG,
    )
    # Items 0 and len(TRAINABLE_EXPORT_FILES) are both MAIN.bmp (sorted order).
    main_spec = TRAINABLE_EXPORT_SPECS[0]
    same_file_indices = [
        i for i in range(len(ds)) if ds.items[i][1] == main_spec.file_name
    ]
    assert len(same_file_indices) == 2

    class _Subset(torch.utils.data.Dataset):
        def __init__(self, base, idx):
            self.base = base; self.idx = idx
        def __len__(self): return len(self.idx)
        def __getitem__(self, i): return self.base[self.idx[i]]

    sub = _Subset(ds, same_file_indices)
    loader = DataLoader(sub, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    assert batch["target_rgb"].shape == (2, 3, main_spec.h, main_spec.w)
    assert batch["observed_mask"].shape == (2, 1, main_spec.h, main_spec.w)
    assert batch["observed_rgb"].shape == (2, 3, main_spec.h, main_spec.w)


def test_missing_bmp_raises(tmp_path):
    skin = tmp_path / "broken"
    skin.mkdir()
    # Only copy a couple of BMPs to simulate an incomplete skin.
    for fname in ["MAIN.bmp", "CBUTTONS.bmp"]:
        shutil.copyfile(DEFAULT_SKIN / fname, skin / fname)
    with pytest.raises(FileNotFoundError, match="missing trainable BMP"):
        V7CompletionDataset(
            skin_sources={"broken": skin},
            state_families_path=CONFIG,
        )


def test_wrong_dimensions_raises(tmp_path):
    skin = tmp_path / "wrong_dim"
    shutil.copytree(DEFAULT_SKIN, skin)
    # Resize one BMP to an unexpected size.
    with Image.open(skin / "MAIN.bmp") as im:
        bad = im.convert("RGB").resize((100, 50))
    bad.save(skin / "MAIN.bmp")
    with pytest.raises(ValueError, match="!="):
        V7CompletionDataset(
            skin_sources={"wrong": skin},
            state_families_path=CONFIG,
        )


def test_unknown_file_in_provenance_pools_raises():
    pools = {"NOT_A_FILE.bmp": [np.ones((10, 10), dtype=np.uint8)]}
    with pytest.raises(ValueError, match="unknown file"):
        V7CompletionDataset(
            skin_sources={"default": DEFAULT_SKIN},
            state_families_path=CONFIG,
            provenance_pools=pools,
        )


def test_provenance_pool_shape_mismatch_raises():
    spec = TRAINABLE_EXPORT_SPECS[0]  # MAIN.bmp
    pools = {spec.file_name: [np.ones((spec.h + 1, spec.w), dtype=np.uint8)]}
    with pytest.raises(ValueError, match="expected"):
        V7CompletionDataset(
            skin_sources={"default": DEFAULT_SKIN},
            state_families_path=CONFIG,
            provenance_pools=pools,
        )


def test_empty_skin_sources_raises():
    with pytest.raises(ValueError, match="at least one skin"):
        V7CompletionDataset(
            skin_sources={},
            state_families_path=CONFIG,
        )
