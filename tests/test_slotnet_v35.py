"""Tests for SlotNetV3.5 exported BMP prediction."""
from __future__ import annotations

import torch

from atlas_ai.export_spec import TRAINABLE_EXPORT_SPECS
from models.slotnet_v35 import SlotNetV35


def _build_model() -> SlotNetV35:
    return SlotNetV35(base_channels=4, style_dim=32, head_channels=8)


def test_v35_forward_returns_exact_exported_bmp_tensors():
    model = _build_model().cpu()
    out = model(torch.zeros(1, 3, 32, 32))
    assert set(out) == {"files"}
    assert set(out["files"]) == {spec.file_name for spec in TRAINABLE_EXPORT_SPECS}
    for spec in TRAINABLE_EXPORT_SPECS:
        assert out["files"][spec.file_name].shape == (1, 3, spec.h, spec.w)


def test_v35_has_no_prior_observed_or_full_atlas_prediction():
    model = _build_model().cpu()
    out = model(torch.zeros(1, 3, 32, 32))
    assert "prediction" not in out
    assert "prior_rgb" not in out
    assert "observed_logits" not in out
    assert "default_atlas" not in model.state_dict()
    assert not any(key.startswith("observed_head") for key in model.state_dict())


def test_v35_has_version_marker_buffer():
    model = _build_model().cpu()
    assert int(model.slotnet_version.item()) == 35


def test_v35_backward_runs():
    model = _build_model().cpu()
    out = model(torch.zeros(1, 3, 32, 32))
    loss = sum(tensor.mean() for tensor in out["files"].values())
    loss.backward()
