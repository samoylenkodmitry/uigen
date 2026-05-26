"""V7.1 StateFamilyExpander: families, dataset pairs, sampler, model."""

from __future__ import annotations

from pathlib import Path

import torch

from collections import Counter

import pytest

from atlas_ai.state_pairs_dataset import StatePairsDataset, collect_alt_families
from atlas_ai.v7_batching import SameKeyBatchSampler, WeightedSameKeyBatchSampler
from models.v7_state_expander import V7StateExpander


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/state_families_classic.yaml"
DEFAULT_SKIN = ROOT / "assets/default_skin"


def test_collect_alt_families_excludes_components():
    fams = collect_alt_families(CONFIG)
    keyed = {(f.file_name, f.family) for f in fams}
    # Alternatives we expect to be present.
    assert ("VOLUME.bmp", "slider_frames") in keyed
    assert ("BALANCE.bmp", "slider_frames") in keyed
    assert ("EQMAIN.bmp", "slider_frames") in keyed
    assert ("EQMAIN.bmp", "on_button") in keyed
    assert ("CBUTTONS.bmp", "play") in keyed
    assert ("SHUFREP.bmp", "repeat") in keyed
    assert ("MONOSTER.bmp", "indicators") in keyed
    assert ("PLAYPAUS.bmp", "status") in keyed
    # Components / single must be excluded.
    assert ("POSBAR.bmp", "parts") not in keyed
    assert ("MAIN.bmp", "panel") not in keyed
    assert ("TITLEBAR.bmp", "corner_buttons") not in keyed
    assert ("PLEDIT.bmp", "footer") not in keyed
    assert ("SHUFREP.bmp", "pl_toggle") not in keyed  # single
    assert ("VOLUME.bmp", "thumb") not in keyed        # components


def test_alt_families_have_uniform_rect_size_and_stable_ids():
    fams = collect_alt_families(CONFIG)
    for i, f in enumerate(fams):
        assert f.family_id == i  # id is position in the stable list
        sizes = {(r.w, r.h) for r in f.rects}
        assert len(sizes) == 1, (f.file_name, f.family, sizes)
        assert f.num_frames >= 2
    # VOLUME slider has 28 frames -> drives max_frames.
    vol = next(f for f in fams if (f.file_name, f.family) == ("VOLUME.bmp", "slider_frames"))
    assert vol.num_frames == 28


def test_dataset_pairs_and_identity_invariant():
    ds = StatePairsDataset({"default": str(DEFAULT_SKIN)}, CONFIG, include_identity=True)
    # item count == sum of N^2 over families (one skin).
    expected = sum(f.num_frames ** 2 for f in ds.alt_families)
    assert len(ds) == expected
    assert ds.max_frames == 28
    # find an identity pair and an off-diagonal pair for VOLUME slider_frames.
    vol_id = next(f.family_id for f in ds.alt_families
                  if (f.file_name, f.family) == ("VOLUME.bmp", "slider_frames"))
    ident_idx = next(i for i, (s, fid, a, b) in enumerate(ds.items)
                     if fid == vol_id and a == b == 3)
    off_idx = next(i for i, (s, fid, a, b) in enumerate(ds.items)
                   if fid == vol_id and a == 3 and b == 12)
    it_id = ds[ident_idx]
    it_off = ds[off_idx]
    # identity pair: source == target exactly.
    assert torch.equal(it_id["source_rgb"], it_id["target_rgb"])
    assert it_id["is_identity"] == 1
    # off-diagonal: shapes match (same family), content differs (slider moved).
    assert it_off["source_rgb"].shape == it_off["target_rgb"].shape == (3, 13, 68)
    assert it_off["is_identity"] == 0
    assert not torch.equal(it_off["source_rgb"], it_off["target_rgb"])
    assert it_off["target_support"].shape == (1, 13, 68)


def test_group_keys_match_items_and_batch_size():
    ds = StatePairsDataset({"default": str(DEFAULT_SKIN)}, CONFIG)
    assert len(ds.group_keys) == len(ds)
    # group key is (file, family); a VOLUME item maps to VOLUME slider key.
    fid = next(f.family_id for f in ds.alt_families
               if (f.file_name, f.family) == ("VOLUME.bmp", "slider_frames"))
    i = next(i for i, (s, f, a, b) in enumerate(ds.items) if f == fid)
    assert ds.group_keys[i] == "VOLUME/slider_frames"


def test_samekey_sampler_batches_share_key_and_cover_all():
    keys = ["a", "a", "a", "b", "b", "c"]
    sampler = SameKeyBatchSampler(keys, batch_size=2, shuffle=False)
    seen: set[int] = set()
    for batch in sampler:
        batch_keys = {keys[i] for i in batch}
        assert len(batch_keys) == 1, batch_keys  # one key per batch
        seen.update(batch)
    assert seen == set(range(len(keys)))  # every index covered


def test_family_keys_are_unique_across_files():
    """'slider_frames' appears in VOLUME/BALANCE/EQMAIN — keys must not collide."""
    fams = collect_alt_families(CONFIG)
    keys = [f.key for f in fams]
    assert len(keys) == len(set(keys))
    assert "VOLUME/slider_frames" in keys
    assert "BALANCE/slider_frames" in keys
    assert "EQMAIN/slider_frames" in keys


def test_weighted_samekey_balances_unequal_groups():
    """Equal key weights -> a 4-item group gets ~as many batches as a 100-item
    group (fixes the pair-count exposure bias)."""
    keys = ["big"] * 100 + ["small"] * 4
    gen = torch.Generator().manual_seed(0)
    sampler = WeightedSameKeyBatchSampler(keys, batch_size=3, num_batches=600, generator=gen)
    counts: Counter[str] = Counter()
    for batch in sampler:
        bk = {keys[i] for i in batch}
        assert len(bk) == 1  # same-key batch
        counts[bk.pop()] += 1
    assert 0.4 < counts["small"] / 600 < 0.6  # balanced despite 4 vs 100 items


def test_weighted_samekey_item_weights_downweight_identity():
    keys = ["k", "k"]
    item_weights = [0.1, 1.0]  # index 0 is the "identity" pair
    gen = torch.Generator().manual_seed(0)
    sampler = WeightedSameKeyBatchSampler(keys, batch_size=1, num_batches=3000,
                                          item_weights=item_weights, generator=gen)
    counts: Counter[int] = Counter()
    for batch in sampler:
        counts[batch[0]] += 1
    assert counts[0] / 3000 < 0.2   # identity ~ 0.1/1.1 ≈ 9%
    assert counts[1] / 3000 > 0.8


def test_model_forward_shapes_and_range():
    model = V7StateExpander(num_families=16, max_frames=28, base_channels=8,
                            file_embedding_dim=4, family_embedding_dim=4, frame_embedding_dim=4)
    src = torch.rand(2, 3, 13, 68)
    with torch.no_grad():
        out = model(src, torch.tensor([3, 5]), torch.tensor([12, 5]),
                    torch.tensor([0, 1]), torch.tensor([9, 9]))
    assert out.shape == (2, 3, 13, 68)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_output_modes_forward_and_range():
    args = dict(num_families=16, max_frames=28, base_channels=8,
                file_embedding_dim=4, family_embedding_dim=4, frame_embedding_dim=4)
    src = torch.rand(2, 3, 9, 9)
    for mode in ("residual", "direct", "unbounded", "gated"):
        m = V7StateExpander(output_mode=mode, **args)
        assert m.output_mode == mode
        assert int(m.output_mode_buffer.item()) in (0, 1, 2, 3)
        idx = (torch.tensor([0, 1]), torch.tensor([1, 0]), torch.tensor([0, 1]), torch.tensor([9, 9]))
        with torch.no_grad():
            out = m(src, *idx)
            assert out.shape == (2, 3, 9, 9)
            assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
            if mode == "gated":
                # gate head returns ([0,1] gate, logits) when requested, and
                # inits near copy (bias -2.0 -> gate well below 0.5).
                out2, gate, gate_logits = m(src, *idx, return_gate=True)
                assert gate.shape == gate_logits.shape == (2, 1, 9, 9)
                assert float(gate.mean()) < 0.3
    with pytest.raises(ValueError):
        V7StateExpander(output_mode="bogus", **args)


def test_compute_loss_gated_path_runs():
    """Guards the trainer's gated branch (forward call must pass family_id)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("tr_se", ROOT / "train_v7_state_expander.py")
    tr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tr)
    model = V7StateExpander(num_families=16, max_frames=28, base_channels=8,
                            file_embedding_dim=4, family_embedding_dim=4,
                            frame_embedding_dim=4, output_mode="gated")
    b = 2
    batch = {
        "source_rgb": torch.rand(b, 3, 9, 9), "target_rgb": torch.rand(b, 3, 9, 9),
        "target_support": torch.ones(b, 1, 9, 9),
        "source_idx": torch.tensor([0, 1]), "target_idx": torch.tensor([1, 0]),
        "family_id": torch.tensor([0, 1]), "file_id": torch.tensor([9, 9]),
        "skin_index": torch.tensor([0, 0]),
    }
    out = tr.compute_loss(model, batch, {}, torch.device("cpu"), sobel_weight=0.25)
    assert {"total", "total_per_item", "gate_mean", "gate_changed", "gate_unchanged"} <= set(out)
    # With gate supervision on, a gate_loss term appears and total stays finite.
    out2 = tr.compute_loss(model, batch, {}, torch.device("cpu"), sobel_weight=0.25,
                           gate_loss_weight=0.05, gate_change_threshold=0.02)
    assert "gate_loss" in out2 and torch.isfinite(out2["total"])


def test_model_forward_with_skin_embedding():
    model = V7StateExpander(num_families=16, max_frames=28, base_channels=8,
                            file_embedding_dim=4, family_embedding_dim=4, frame_embedding_dim=4,
                            num_skins=14, skin_embedding_dim=8)
    src = torch.rand(2, 3, 9, 9)
    out = model(src, torch.tensor([0, 1]), torch.tensor([2, 0]),
                torch.tensor([5, 5]), torch.tensor([7, 7]),
                skin_id=torch.tensor([0, 13]))
    assert out.shape == (2, 3, 9, 9)
    # skin_id required when num_skins>0.
    try:
        model(src, torch.tensor([0, 1]), torch.tensor([2, 0]),
              torch.tensor([5, 5]), torch.tensor([7, 7]))
        assert False, "expected ValueError without skin_id"
    except ValueError:
        pass
