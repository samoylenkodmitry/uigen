"""Tests for SlotNetV3.4 direct RGB atlas prediction."""
from __future__ import annotations

import torch

from atlas_ai.profiles import load_atlas_profile
from models.slotnet_v34 import SlotNetV34


def _build_model(base_channels: int = 8) -> SlotNetV34:
    return SlotNetV34(
        atlas_profile=load_atlas_profile("configs/atlas_train_v1.json"),
        base_channels=base_channels,
    )


def test_v34_forward_returns_rgb_atlas_prediction():
    model = _build_model().cpu()
    out = model(torch.zeros(1, 3, 64, 64))
    assert out["prediction"].shape == (1, 3, 1024, 1024)


def test_v34_has_no_prior_or_observed_head():
    model = _build_model().cpu()
    out = model(torch.zeros(1, 3, 64, 64))
    assert "prior_rgb" not in out
    assert "observed_logits" not in out
    assert "default_atlas" not in model.state_dict()
    assert not any(key.startswith("observed_head") for key in model.state_dict())


def test_v34_has_version_marker_buffer():
    model = _build_model().cpu()
    assert int(model.slotnet_version.item()) == 34


def test_v34_backward_runs():
    model = _build_model().cpu()
    loss = model(torch.zeros(1, 3, 64, 64))["prediction"].mean()
    loss.backward()
