"""V7 support-masked losses and metrics."""

from __future__ import annotations

import torch

from models.losses_v7 import (
    hidden_supported_hit5,
    hidden_supported_l1_loss,
    hidden_supported_l1_per_item,
    observed_passthrough_mae,
    support_masked_hit5,
    support_masked_l1_loss,
    support_masked_sobel_mae,
)


def test_hit5_stays_fraction_for_batch_size_greater_than_one():
    pred = torch.zeros(2, 3, 4, 4)
    target = torch.zeros_like(pred)
    support = torch.ones(4, 4, dtype=torch.bool)

    assert support_masked_hit5(pred, target, support).item() == 1.0

    pred[0, :, 0, 0] = 1.0
    got = support_masked_hit5(pred, target, support).item()
    assert 0.0 <= got <= 1.0
    assert got == (31 / 32)


def test_l1_ignores_unsupported_pixels():
    pred = torch.zeros(1, 3, 2, 2)
    target = torch.zeros_like(pred)
    target[:, :, 0, 0] = 1.0
    support = torch.tensor([[False, True], [True, True]])
    assert support_masked_l1_loss(pred, target, support).item() == 0.0


def test_sobel_accepts_batch_support_mask():
    pred = torch.zeros(2, 3, 4, 4)
    target = torch.zeros_like(pred)
    support = torch.ones(2, 1, 4, 4, dtype=torch.bool)
    got = support_masked_sobel_mae(pred, target, support)
    assert torch.isfinite(got)
    assert got.item() == 0.0


# ---------------------------------------------------------------------------
# Hidden-normalized losses (Task 1). hidden = (1 - observed) * support.
# ---------------------------------------------------------------------------


def test_all_observed_has_zero_hidden_and_no_dilution():
    """Case 1: an all-observed mask has an empty hidden denominator, so the
    hidden loss is exactly 0 — never a diluted small number."""
    target = torch.zeros(1, 3, 4, 4)
    final = target.clone()
    final[:, :, 0, 0] = 1.0  # error injected on an *observed* pixel
    observed = torch.ones(1, 1, 4, 4)
    support = torch.ones(4, 4)
    assert hidden_supported_l1_loss(final, target, observed, support).item() == 0.0
    per_item, has_hidden = hidden_supported_l1_per_item(final, target, observed, support)
    assert per_item.shape == (1,)
    assert has_hidden.tolist() == [False]
    # The observed-passthrough diagnostic still sees the injected error.
    assert observed_passthrough_mae(final, target, observed, support).item() > 0.0


def test_hidden_normalizes_by_hidden_pixels_only():
    """Case 2: a half-hidden sample normalizes by hidden pixels only."""
    target = torch.zeros(1, 3, 4, 4)
    final = torch.zeros(1, 3, 4, 4)
    observed = torch.ones(1, 1, 4, 4)
    observed[:, :, 0:2, 0:2] = 0.0   # 4 hidden pixels
    final[:, :, 0:2, 0:2] = 0.2      # error only on hidden pixels
    support = torch.ones(4, 4)
    # mean over the 4 hidden pixels == 0.2, regardless of the 12 observed ones.
    assert abs(hidden_supported_l1_loss(final, target, observed, support).item() - 0.2) < 1e-6
    # full-supported divides the same error by all 16 supported pixels.
    full = support_masked_l1_loss(final, target, support).item()
    assert abs(full - 0.2 * 4 / 16) < 1e-6


def test_hard_copied_observed_does_not_flatter_hidden_mae():
    """Case 3: adding more perfectly-copied observed pixels shrinks the
    full-supported MAE but leaves hidden MAE untouched."""
    def measure(width: int) -> tuple[float, float]:
        target = torch.zeros(1, 3, 8, width)
        final = torch.zeros(1, 3, 8, width)
        observed = torch.ones(1, 1, 8, width)
        observed[:, :, 0:2, 0:2] = 0.0   # same 4 hidden pixels either way
        final[:, :, 0:2, 0:2] = 0.3
        support = torch.ones(8, width)
        return (
            hidden_supported_l1_loss(final, target, observed, support).item(),
            support_masked_l1_loss(final, target, support).item(),
        )

    h_small, f_small = measure(8)
    h_big, f_big = measure(64)
    assert abs(h_small - 0.3) < 1e-6
    assert abs(h_big - 0.3) < 1e-6   # hidden MAE unchanged by extra observed
    assert f_big < f_small           # full MAE diluted by extra observed


def test_hidden_hit5_denominator_is_hidden_count():
    """Case 4: hidden_hit5 counts hidden pixels, not all supported pixels."""
    target = torch.zeros(1, 3, 4, 4)
    final = torch.zeros(1, 3, 4, 4)
    observed = torch.ones(1, 1, 4, 4)
    observed[:, :, 0:2, :] = 0.0   # top half (8 px) hidden
    final[:, :, 0:2, :] = 1.0      # hidden pixels all miss; observed all exact
    support = torch.ones(4, 4)
    assert hidden_supported_hit5(final, target, observed, support).item() == 0.0
    # full-supported hit5 still credits the 8 exact observed pixels -> 0.5.
    assert abs(support_masked_hit5(final, target, support).item() - 0.5) < 1e-6


def test_per_item_hidden_metrics_for_same_file_batch():
    """Case 5: per-item hidden metrics work across a same-file batch with one
    shared [H, W] support mask and differing observed masks."""
    support = torch.ones(4, 4)
    target = torch.zeros(3, 3, 4, 4)
    final = torch.zeros(3, 3, 4, 4)
    observed = torch.ones(3, 1, 4, 4)
    # item 0: fully observed -> no hidden pixels.
    # item 1: hide a 2x2 block, error 0.2.
    observed[1, :, 0:2, 0:2] = 0.0
    final[1, :, 0:2, 0:2] = 0.2
    # item 2: hide the top row, error 0.4.
    observed[2, :, 0, :] = 0.0
    final[2, :, 0, :] = 0.4
    per_item, has_hidden = hidden_supported_l1_per_item(final, target, observed, support)
    assert per_item.shape == (3,)
    assert has_hidden.tolist() == [False, True, True]
    assert per_item[0].item() == 0.0
    assert abs(per_item[1].item() - 0.2) < 1e-6
    assert abs(per_item[2].item() - 0.4) < 1e-6
