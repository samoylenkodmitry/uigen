"""V7 observed-mask generator: shape, mode mix, family semantics, oracle."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from atlas_ai.export_spec import TRAINABLE_EXPORT_SPECS
from atlas_ai.state_families import StateRect, group_by_family, load_state_families
from atlas_ai.v7_masks import (
    MASK_MODES,
    V7MaskWeights,
    apply_observed_mask,
    make_provenance_mask,
    make_random_rect_mask,
    make_state_family_mask,
    make_whole_file_mask,
    sample_v7_observed_mask,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/state_families_classic.yaml"


def test_random_rect_mask_zeros_a_rectangle():
    rng = np.random.default_rng(0)
    mask = make_random_rect_mask(rng, 32, 64)
    assert mask.shape == (32, 64)
    assert mask.dtype == np.uint8
    assert mask.min() == 0 and mask.max() == 1
    # At least one zero (the rect) and at least some ones (everywhere else).
    assert 0 < mask.sum() < mask.size


def test_whole_file_mask_zeros():
    mask = make_whole_file_mask(8, 16)
    assert mask.shape == (8, 16)
    assert mask.sum() == 0


def test_provenance_mask_rejects_empty_pool():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="empty"):
        make_provenance_mask(rng, [], 4, 4)


def test_provenance_mask_returns_one_of_pool():
    rng = np.random.default_rng(0)
    pool = [
        np.zeros((4, 4), dtype=np.uint8),
        np.ones((4, 4), dtype=np.uint8),
    ]
    seen = {0, 0}
    for _ in range(50):
        mask = make_provenance_mask(rng, pool, 4, 4)
        seen.add(int(mask.sum()))
    # Both members of the pool should appear over 50 draws.
    assert {0, 16} <= seen


def test_state_family_mask_reveals_one_hides_siblings():
    families = load_state_families(CONFIG)
    rects = families["VOLUME.bmp"]
    spec = next(s for s in TRAINABLE_EXPORT_SPECS if s.file_name == "VOLUME.bmp")
    rng = np.random.default_rng(0)
    # Run many draws; check that exactly one of the slider frames is revealed
    # while the others are hidden, every time.
    grouped = group_by_family(rects)
    frame_rects = grouped["slider_frames"]
    assert len(frame_rects) == 28
    revealed_counts: Counter[int] = Counter()
    for _ in range(200):
        mask = make_state_family_mask(rng, rects, spec.h, spec.w)
        # In the slider_frames family rectangles, exactly one rect should be
        # fully visible (mask=1 inside) and the other 27 fully hidden.
        revealed = [
            i for i, r in enumerate(frame_rects)
            if mask[r.y : r.y + r.h, r.x : r.x + r.w].all()
        ]
        hidden = [
            i for i, r in enumerate(frame_rects)
            if (mask[r.y : r.y + r.h, r.x : r.x + r.w] == 0).all()
        ]
        # When the chosen family is 'slider_frames', exactly one is revealed
        # and the other 27 hidden. When the chosen family is 'thumb', no
        # slider frames are explicitly hidden -> all 28 stay revealed.
        if len(revealed) == 28:
            continue  # 'thumb' family chosen this draw
        assert len(revealed) == 1
        assert len(hidden) == 27
        revealed_counts[revealed[0]] += 1
    # Every frame index should be selected at least once over 200 draws.
    assert len(revealed_counts) >= 25


def test_eqmain_state_family_mask_hides_both_slider_rows():
    """EQMAIN's 28 slider frames span two rows but are one state family."""
    families = load_state_families(CONFIG)
    rects = families["EQMAIN.bmp"]
    spec = next(s for s in TRAINABLE_EXPORT_SPECS if s.file_name == "EQMAIN.bmp")
    grouped = group_by_family(rects)
    frames = grouped["slider_frames"]
    assert len(frames) == 28
    rng = np.random.default_rng(4)
    seen_hidden_across_rows = False
    for _ in range(300):
        mask = make_state_family_mask(rng, rects, spec.h, spec.w)
        revealed = [
            i for i, r in enumerate(frames)
            if mask[r.y : r.y + r.h, r.x : r.x + r.w].all()
        ]
        hidden = [
            i for i, r in enumerate(frames)
            if (mask[r.y : r.y + r.h, r.x : r.x + r.w] == 0).all()
        ]
        if len(revealed) == 1:
            assert len(hidden) == 27
            hidden_rows = {0 if i < 14 else 1 for i in hidden}
            seen_hidden_across_rows = seen_hidden_across_rows or hidden_rows == {0, 1}
    assert seen_hidden_across_rows


def test_state_family_mask_falls_back_when_no_eligible_family():
    """A single one-rect family contributes no hides; mask stays all-ones."""
    rng = np.random.default_rng(0)
    rects = [StateRect("MAIN.bmp", "panel", "full", 0, 0, 275, 115)]
    mask = make_state_family_mask(rng, rects, 115, 275)
    assert mask.shape == (115, 275)
    assert mask.sum() == 115 * 275


def test_sample_v7_distribution_matches_weights_when_available():
    """With all modes available, draws should track the weight mix."""
    families = load_state_families(CONFIG)
    rects = families["VOLUME.bmp"]
    pool = [np.ones((115, 275), dtype=np.uint8)] * 4
    # Shape pool to match VOLUME, not MAIN, so provenance-mode mask shape
    # matches the requested (h, w).
    pool = [np.ones((433, 68), dtype=np.uint8) for _ in range(4)]

    rng = np.random.default_rng(123)
    weights = V7MaskWeights()  # plan defaults
    n = 4000
    counts: Counter[str] = Counter()
    for _ in range(n):
        _, mode = sample_v7_observed_mask(
            rng, h=433, w=68, family_rects=rects, visible_masks=pool, weights=weights,
        )
        counts[mode] += 1
    expected = {
        "provenance": weights.provenance,
        "state_family": weights.state_family,
        "random_rect": weights.random_rect,
        "whole_file": weights.whole_file,
    }
    for mode in MASK_MODES:
        observed_frac = counts[mode] / n
        assert abs(observed_frac - expected[mode]) < 0.04, (
            f"{mode}: expected {expected[mode]:.2f} got {observed_frac:.3f}"
        )


def test_sample_v7_redistributes_when_no_provenance():
    families = load_state_families(CONFIG)
    rects = families["VOLUME.bmp"]
    rng = np.random.default_rng(0)
    n = 2000
    counts: Counter[str] = Counter()
    for _ in range(n):
        _, mode = sample_v7_observed_mask(
            rng, h=433, w=68, family_rects=rects, visible_masks=None,
        )
        counts[mode] += 1
    assert counts["provenance"] == 0
    # All other modes get a positive share.
    assert counts["state_family"] > 0
    assert counts["random_rect"] > 0
    assert counts["whole_file"] > 0


def test_apply_observed_mask_zeros_hidden_pixels():
    target = np.full((4, 4, 3), 0.5, dtype=np.float32)
    mask = np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 1, 1],
    ], dtype=np.uint8)
    out = apply_observed_mask(target, mask)
    assert out.shape == (4, 4, 3)
    assert (out[mask == 0] == 0).all()
    assert (out[mask == 1] == 0.5).all()


def test_apply_observed_mask_shape_check():
    with pytest.raises(ValueError, match="shape"):
        apply_observed_mask(np.zeros((4, 4, 3), dtype=np.float32), np.zeros((3, 4), dtype=np.uint8))


def test_oracle_observed_target_is_exact_loss_zero():
    """When observed_mask == 1 everywhere, the 'observed' input already IS
    the target; a learnable completer should be able to reach loss 0 by
    learning identity on the masked input. We verify the limit case directly:
    if the completer outputs observed_rgb unchanged, masked L1 over observed
    pixels is 0."""
    target = np.random.default_rng(0).uniform(0, 1, size=(32, 32, 3)).astype(np.float32)
    mask = np.ones((32, 32), dtype=np.uint8)
    observed = apply_observed_mask(target, mask)
    diff = np.abs(observed - target)
    assert diff.max() == 0.0


def test_oracle_random_mask_completer_with_known_inverse():
    """If we *know* the target, we can construct an inverse: copy observed
    pixels through, fill hidden pixels with the target. Loss must be 0.
    This is the oracle-completer sanity that the V7 plan's Stage 1 calls for."""
    rng = np.random.default_rng(5)
    families = load_state_families(CONFIG)
    rects = families["VOLUME.bmp"]
    spec = next(s for s in TRAINABLE_EXPORT_SPECS if s.file_name == "VOLUME.bmp")
    target = rng.uniform(0, 1, size=(spec.h, spec.w, 3)).astype(np.float32)
    mask, _ = sample_v7_observed_mask(
        rng, h=spec.h, w=spec.w, family_rects=rects, visible_masks=None,
    )
    observed = apply_observed_mask(target, mask)
    # Oracle inverse: take observed pixels where mask=1, target pixels where mask=0.
    oracle_completion = np.where(mask[..., None].astype(bool), observed, target)
    loss = float(np.abs(oracle_completion - target).mean())
    assert loss == 0.0
