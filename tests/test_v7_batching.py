"""SameFileBatchSampler: same-file guarantee, coverage, determinism."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from collections import Counter

from atlas_ai.dataset_v7_completion import V7CompletionDataset
from atlas_ai.export_spec import TRAINABLE_EXPORT_FILES, TRAINABLE_EXPORT_SPECS
from atlas_ai.v7_batching import SameFileBatchSampler, WeightedSameFileBatchSampler


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


def test_weighted_sampler_never_mixes_file_shapes():
    items = _items_for_n_skins(3)
    weights = {f: 1.0 for f in TRAINABLE_EXPORT_FILES}
    sampler = WeightedSameFileBatchSampler(
        items, batch_size=2, file_weights=weights, num_batches=100,
        generator=torch.Generator().manual_seed(0),
    )
    for batch in sampler:
        files = {items[idx][1] for idx in batch}
        assert len(files) == 1, files


def test_weighted_sampler_emits_exactly_num_batches():
    items = _items_for_n_skins(2)
    weights = {f: 1.0 for f in TRAINABLE_EXPORT_FILES}
    sampler = WeightedSameFileBatchSampler(
        items, batch_size=1, file_weights=weights, num_batches=37,
        generator=torch.Generator().manual_seed(0),
    )
    assert len(sampler) == 37
    assert sum(1 for _ in sampler) == 37


def test_weighted_sampler_distribution_matches_weights():
    items = _items_for_n_skins(3)
    weights = {
        "MAIN.bmp":     4.0,
        "EQMAIN.bmp":   8.0,
        "PLAYPAUS.bmp": 1.0,
        # the rest have weight 0 -> skipped
    }
    n = 5000
    sampler = WeightedSameFileBatchSampler(
        items, batch_size=1, file_weights=weights, num_batches=n,
        generator=torch.Generator().manual_seed(42),
    )
    counts: Counter[str] = Counter()
    for batch in sampler:
        counts[items[batch[0]][1]] += 1
    # Only the three weighted files should ever appear.
    assert set(counts) == {"MAIN.bmp", "EQMAIN.bmp", "PLAYPAUS.bmp"}
    # Empirical fractions within 3% of normalized weights.
    total_w = 4.0 + 8.0 + 1.0
    expected = {
        "MAIN.bmp": 4.0 / total_w,
        "EQMAIN.bmp": 8.0 / total_w,
        "PLAYPAUS.bmp": 1.0 / total_w,
    }
    for file_name, want in expected.items():
        got = counts[file_name] / n
        assert abs(got - want) < 0.03, f"{file_name}: want {want:.3f} got {got:.3f}"


def test_weighted_sampler_one_skin_replacement_works():
    """With one skin, every file has exactly one item. Weighted sampling
    with replacement still yields batch_size > 1 from that one item."""
    items = _items_for_n_skins(1)
    weights = {"EQMAIN.bmp": 1.0}
    sampler = WeightedSameFileBatchSampler(
        items, batch_size=4, file_weights=weights, num_batches=5,
        generator=torch.Generator().manual_seed(0),
    )
    expected_index = next(i for i, (_, f) in enumerate(items) if f == "EQMAIN.bmp")
    for batch in sampler:
        assert len(batch) == 4
        assert all(idx == expected_index for idx in batch)


def test_weighted_sampler_rejects_no_positive_weights():
    items = _items_for_n_skins(1)
    weights = {f: 0.0 for f in TRAINABLE_EXPORT_FILES}
    with pytest.raises(ValueError, match="positive weight"):
        WeightedSameFileBatchSampler(
            items, batch_size=1, file_weights=weights, num_batches=10,
        )


def test_weighted_sampler_rejects_invalid_args():
    items = _items_for_n_skins(1)
    weights = {"EQMAIN.bmp": 1.0}
    with pytest.raises(ValueError):
        WeightedSameFileBatchSampler(items, batch_size=0,
                                     file_weights=weights, num_batches=10)
    with pytest.raises(ValueError):
        WeightedSameFileBatchSampler(items, batch_size=1,
                                     file_weights=weights, num_batches=0)


def test_weighted_sampler_with_real_dataloader():
    """End-to-end: weighted sampler drives a real DataLoader on V7Dataset."""
    dataset = V7CompletionDataset(
        skin_sources={"a": DEFAULT_SKIN},
        state_families_path=CONFIG,
    )
    weights = {"EQMAIN.bmp": 8.0, "MAIN.bmp": 4.0, "PLAYPAUS.bmp": 1.0}
    sampler = WeightedSameFileBatchSampler(
        dataset.items, batch_size=1, file_weights=weights, num_batches=20,
        generator=torch.Generator().manual_seed(0),
    )
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
    files_seen: Counter[str] = Counter()
    for batch in loader:
        assert len(set(batch["file_name"])) == 1
        files_seen[batch["file_name"][0]] += 1
    assert set(files_seen) <= set(weights)
    assert sum(files_seen.values()) == 20


def test_weighted_sampler_without_replacement_gives_distinct_indices():
    """When the file group is at least batch_size, within_file_replacement=False
    must draw distinct indices within a batch — that is the whole point of the
    flag for Gate B (one batch should cover distinct skins, not duplicates)."""
    # 14 "skins" so every file group has 14 items.
    items = _items_for_n_skins(14)
    weights = {"EQMAIN.bmp": 1.0}
    sampler = WeightedSameFileBatchSampler(
        items, batch_size=8, file_weights=weights, num_batches=20,
        generator=torch.Generator().manual_seed(0),
        within_file_replacement=False,
    )
    for batch in sampler:
        assert len(batch) == 8
        assert len(set(batch)) == 8, batch  # distinct dataset indices => distinct skins


def test_weighted_sampler_without_replacement_falls_back_when_group_too_small():
    """Safety: if the chosen file group has fewer items than batch_size, the
    sampler must still yield a batch (with replacement for that file), not
    raise — otherwise one-skin runs would explode."""
    items = _items_for_n_skins(1)  # group size 1 per file
    weights = {"EQMAIN.bmp": 1.0}
    sampler = WeightedSameFileBatchSampler(
        items, batch_size=4, file_weights=weights, num_batches=5,
        generator=torch.Generator().manual_seed(0),
        within_file_replacement=False,
    )
    for batch in sampler:
        assert len(batch) == 4  # no crash; replacement used because group<batch


def test_build_file_weights_replace_only_samples_listed_files(tmp_path):
    """In replace mode, a YAML listing only BALANCE/VOLUME must produce a
    weights dict that, when handed to the sampler, draws batches only from
    those two file groups — never from defaults. Without this, a yaml named
    'strip-only' silently inherits EQMAIN/MAIN/etc and the probe is not
    actually focused.
    """
    from train_v7_completer import build_file_weights

    yaml_path = tmp_path / "bv_only.yaml"
    yaml_path.write_text("BALANCE.bmp: 1\nVOLUME.bmp: 1\n", encoding="utf-8")
    weights = build_file_weights(yaml_path, mode="replace")
    assert set(weights) == {"BALANCE.bmp", "VOLUME.bmp"}

    items = _items_for_n_skins(8)
    sampler = WeightedSameFileBatchSampler(
        items, batch_size=4, file_weights=weights, num_batches=200,
        generator=torch.Generator().manual_seed(0),
    )
    seen_files: set[str] = set()
    for batch in sampler:
        for idx in batch:
            seen_files.add(items[idx][1])
    assert seen_files == {"BALANCE.bmp", "VOLUME.bmp"}, (
        f"replace mode should restrict to listed files, got {seen_files}"
    )


def test_build_file_weights_merge_keeps_defaults(tmp_path):
    """In merge mode (default, backward compat), a YAML override must not
    drop the other files — they should remain at their default weights.
    This protects the older multi-file recipes from accidentally narrowing.
    """
    from train_v7_completer import build_file_weights, DEFAULT_FILE_WEIGHTS

    yaml_path = tmp_path / "doubled_strips.yaml"
    yaml_path.write_text("BALANCE.bmp: 8\nVOLUME.bmp: 10\n", encoding="utf-8")
    weights = build_file_weights(yaml_path, mode="merge")
    # YAML keys override defaults.
    assert weights["BALANCE.bmp"] == 8
    assert weights["VOLUME.bmp"] == 10
    # Default files NOT in YAML keep their original weight.
    for fn, w in DEFAULT_FILE_WEIGHTS.items():
        if fn not in {"BALANCE.bmp", "VOLUME.bmp"}:
            assert weights[fn] == w, f"merge dropped default for {fn}"


def test_build_file_weights_replace_requires_yaml():
    """Replace mode without a YAML path would zero every file. The helper
    must reject that loudly rather than silently producing an empty dict."""
    from train_v7_completer import build_file_weights

    with pytest.raises(ValueError, match="replace mode requires a YAML path"):
        build_file_weights(None, mode="replace")


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
