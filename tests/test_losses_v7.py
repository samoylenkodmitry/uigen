"""V7 support-masked losses and metrics."""

from __future__ import annotations

import torch

from models.losses_v7 import (
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
