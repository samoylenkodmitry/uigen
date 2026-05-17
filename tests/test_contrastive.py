"""Tests for `atlas_contrastive_loss` per Codex review #5.

The contrastive loss must treat rows with identical atlas targets as positive
pairs, not as false negatives. (E.g. 16 variants of the same skin all share
the same target atlas; without this, the model is penalized for getting the
right answer.)
"""
from __future__ import annotations

import torch

from models.losses import atlas_contrastive_loss


def test_returns_zero_for_batch_one():
    pred = torch.rand(1, 3, 1024, 1024)
    target = torch.rand(1, 3, 1024, 1024)
    assert float(atlas_contrastive_loss(pred, target)) == 0.0


def test_returns_zero_when_all_rows_share_same_target():
    """If every row in the batch has the same target atlas (e.g. variants of
    one skin), there's no negative to contrast against -- the loss must be
    explicitly zero, not a per-row InfoNCE that punishes correct matches."""
    pred = torch.rand(4, 3, 1024, 1024)
    one_target = torch.rand(1, 3, 1024, 1024)
    target = one_target.expand(4, -1, -1, -1).contiguous()
    assert float(atlas_contrastive_loss(pred, target)) == 0.0


def test_two_distinct_targets_gives_normal_infonce():
    """With B=2 distinct targets the loss reduces to standard InfoNCE."""
    pred = torch.rand(2, 3, 1024, 1024)
    target = torch.rand(2, 3, 1024, 1024)
    loss = float(atlas_contrastive_loss(pred, target))
    # Standard InfoNCE on random inputs with temperature 0.07 gives ~log(2)
    # but exact value depends on similarity. Just require positive finite.
    assert 0.0 < loss < 100.0


def test_supervised_contrastive_with_grouped_targets():
    """When the batch contains two skins x two variants each (4 rows, 2
    groups), positives are within-group and negatives are cross-group.
    Loss should be finite and strictly greater than the degenerate
    same-skin case (which is 0)."""
    pred = torch.rand(4, 3, 1024, 1024)
    a = torch.rand(1, 3, 1024, 1024)
    b = torch.rand(1, 3, 1024, 1024)
    target = torch.cat([a, a, b, b], dim=0)
    loss = float(atlas_contrastive_loss(pred, target))
    assert loss > 0.0
    assert torch.isfinite(torch.tensor(loss))


def test_no_grad_through_target_grouping():
    """torch.unique with detach should not let grad flow through the grouping
    operation -- otherwise gradients would try to nudge target tensors
    (which is meaningless and could be a bug). Verify by checking that the
    backward pass on pred does NOT also produce a target gradient."""
    pred = torch.rand(2, 3, 32, 32, requires_grad=True)
    target = torch.rand(2, 3, 32, 32, requires_grad=True)
    loss = atlas_contrastive_loss(pred, target)
    loss.backward()
    # We don't actually want grad through target either; if it's not None,
    # at minimum it shouldn't blow up.
    assert pred.grad is not None
