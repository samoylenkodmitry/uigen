"""SameFileBatchSampler: same-file guarantee, coverage, determinism."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from atlas_ai.dataset_v7_completion import V7CompletionDataset
from atlas_ai.export_spec import TRAINABLE_EXPORT_FILES, TRAINABLE_EXPORT_SPECS
from atlas_ai.v7_batching import SameFileBatchSampler


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/state_families_classic.yaml"
DEFAULT_SKIN = ROOT / "assets/default_skin"


def _items_for_n_skins(n: int) -> list[tuple[str, str]]:
    return [
        (f"skin_{s:02d}", spec.file_name)
        for s in range(n)
        for spec in TRAINABLE_EXPORT_SPECS
    ]


def test_each_batch_has_one_file_name_only():
    items = _items_for_n_skins(4)
    sampler = SameFileBatchSampler(items, batch_size=3, shuffle=True,
                                   generator=torch.Generator().manual_seed(7))
    for batch in sampler:
        file_names = {items[idx][1] for idx in batch}
        assert len(file_names) == 1, file_names


def test_coverage_when_drop_last_false():
    items = _items_for_n_skins(5)
    # batch_size=3, group size=5 -> 2 full + 1 short = 3 batches per file
    sampler = SameFileBatchSampler(items, batch_size=3, shuffle=False)
    seen: set[int] = set()
    for batch in sampler:
        seen.update(batch)
    assert seen == set(range(len(items)))


def test_drop_last_drops_short_batches():
    items = _items_for_n_skins(5)
    sampler = SameFileBatchSampler(items, batch_size=3, shuffle=False, drop_last=True)
    batches = list(sampler)
    # Each file group of 5 -> 1 full batch dropped, so per file: 1 batch.
    assert len(batches) == len(TRAINABLE_EXPORT_FILES)
    for batch in batches:
        assert len(batch) == 3


def test_len_matches_iter():
    items = _items_for_n_skins(5)
    sampler = SameFileBatchSampler(items, batch_size=3, shuffle=False)
    assert len(sampler) == len(list(sampler))


def test_deterministic_with_generator():
    items = _items_for_n_skins(3)
    s1 = SameFileBatchSampler(items, batch_size=2, shuffle=True,
                              generator=torch.Generator().manual_seed(42))
    s2 = SameFileBatchSampler(items, batch_size=2, shuffle=True,
                              generator=torch.Generator().manual_seed(42))
    assert list(s1) == list(s2)


def test_different_seeds_produce_different_orders():
    items = _items_for_n_skins(3)
    s1 = list(SameFileBatchSampler(items, batch_size=2, shuffle=True,
                                   generator=torch.Generator().manual_seed(1)))
    s2 = list(SameFileBatchSampler(items, batch_size=2, shuffle=True,
                                   generator=torch.Generator().manual_seed(2)))
    assert s1 != s2


def test_rejects_invalid_batch_size():
    with pytest.raises(ValueError):
        SameFileBatchSampler(_items_for_n_skins(1), batch_size=0)


def test_real_dataloader_does_not_mix_shapes():
    """Drive a real DataLoader with the sampler; every batch tensor stacks
    cleanly because all items in the batch share file_name -> shape."""
    skin_sources = {"a": DEFAULT_SKIN, "b": DEFAULT_SKIN, "c": DEFAULT_SKIN}
    dataset = V7CompletionDataset(
        skin_sources=skin_sources,
        state_families_path=CONFIG,
    )
    sampler = SameFileBatchSampler(dataset.items, batch_size=2, shuffle=True,
                                   generator=torch.Generator().manual_seed(0))
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
    count = 0
    for batch in loader:
        assert batch["target_rgb"].dim() == 4
        # All file names in the batch agree.
        assert len(set(batch["file_name"])) == 1
        # All tensor shapes match the file's spec.
        file_name = batch["file_name"][0]
        spec = next(s for s in TRAINABLE_EXPORT_SPECS if s.file_name == file_name)
        b = batch["target_rgb"].shape[0]
        assert batch["target_rgb"].shape == (b, 3, spec.h, spec.w)
        assert batch["observed_mask"].shape == (b, 1, spec.h, spec.w)
        assert batch["observed_rgb"].shape == (b, 3, spec.h, spec.w)
        count += 1
    assert count == len(sampler)


def test_default_dataloader_mixed_shapes_explodes_loudly():
    """Sanity that the dangerous path is still loud: default DataLoader with
    batch_size>1 and shuffle=False over the V7 dataset must raise because
    items at consecutive indices belong to different files."""
    dataset = V7CompletionDataset(
        skin_sources={"a": DEFAULT_SKIN},
        state_families_path=CONFIG,
    )
    # The first two items are MAIN.bmp (115x275) and TITLEBAR.bmp (87x344).
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    with pytest.raises(RuntimeError):
        next(iter(loader))
