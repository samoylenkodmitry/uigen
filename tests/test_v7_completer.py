"""V7 completer model: per-file shape, activation, gradient flow, roundtrip."""

from __future__ import annotations

import pytest
import torch

from atlas_ai.export_spec import TRAINABLE_EXPORT_SPECS
from models.v7_completer import V7Completer


FILE_TO_ID = {spec.file_name: idx for idx, spec in enumerate(TRAINABLE_EXPORT_SPECS)}


def _tiny() -> V7Completer:
    return V7Completer(base_channels=8, file_embedding_dim=8, frequencies=(1, 2))


@pytest.mark.parametrize("spec", TRAINABLE_EXPORT_SPECS, ids=lambda s: s.file_name)
def test_forward_shape_per_file(spec):
    model = _tiny().eval()
    observed_rgb = torch.rand(1, 3, spec.h, spec.w)
    observed_mask = torch.zeros(1, 1, spec.h, spec.w)
    file_id = torch.tensor([FILE_TO_ID[spec.file_name]], dtype=torch.long)
    with torch.no_grad():
        out = model(observed_rgb, observed_mask, file_id)
    assert out.shape == (1, 3, spec.h, spec.w)
    assert torch.isfinite(out).all()


def test_output_is_in_unit_range():
    model = _tiny().eval()
    spec = TRAINABLE_EXPORT_SPECS[0]
    obs = torch.rand(2, 3, spec.h, spec.w)
    mask = torch.ones(2, 1, spec.h, spec.w)
    file_id = torch.tensor([0, 0], dtype=torch.long)
    with torch.no_grad():
        out = model(obs, mask, file_id)
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0


def test_gradient_flows_to_encoder_and_output():
    model = _tiny()
    model.train()
    spec = TRAINABLE_EXPORT_SPECS[0]
    obs = torch.rand(1, 3, spec.h, spec.w)
    mask = torch.ones(1, 1, spec.h, spec.w)
    file_id = torch.tensor([0], dtype=torch.long)
    out = model(obs, mask, file_id)
    out.abs().mean().backward()
    assert model.stem[0].weight.grad is not None
    assert model.stem[0].weight.grad.abs().sum() > 0
    assert model.out_proj.weight.grad is not None
    assert model.out_proj.weight.grad.abs().sum() > 0
    # File embedding gradient flows because file_id=0 was used.
    assert model.file_embedding.weight.grad is not None


def test_rejects_wrong_input_shape():
    model = _tiny().eval()
    with pytest.raises(ValueError, match="observed_rgb"):
        model(torch.rand(1, 2, 8, 8), torch.zeros(1, 1, 8, 8), torch.tensor([0]))
    with pytest.raises(ValueError, match="observed_mask"):
        model(torch.rand(1, 3, 8, 8), torch.zeros(1, 2, 8, 8), torch.tensor([0]))


def test_checkpoint_roundtrip(tmp_path):
    from safetensors.torch import load_file, save_file

    model = _tiny().eval()
    spec = TRAINABLE_EXPORT_SPECS[6]  # EQMAIN
    obs = torch.rand(1, 3, spec.h, spec.w)
    mask = torch.ones(1, 1, spec.h, spec.w)
    file_id = torch.tensor([6], dtype=torch.long)
    with torch.no_grad():
        out_a = model(obs, mask, file_id)
    path = tmp_path / "v7.safetensors"
    save_file({k: v.detach().clone() for k, v in model.state_dict().items()}, str(path))
    loaded = load_file(str(path))
    fresh = _tiny().eval()
    fresh.load_state_dict(loaded)
    with torch.no_grad():
        out_b = fresh(obs, mask, file_id)
    assert torch.allclose(out_a, out_b, atol=1e-6)


def test_model_version_buffer_is_70():
    model = _tiny()
    assert int(model.model_version.item()) == 70


def test_supports_smallest_file_playpaus():
    """PLAYPAUS is 42x9 - the smallest trainable file. The 3-level encoder
    must still produce a non-degenerate bottleneck."""
    spec = next(s for s in TRAINABLE_EXPORT_SPECS if s.file_name == "PLAYPAUS.bmp")
    assert (spec.h, spec.w) == (9, 42)
    model = _tiny().eval()
    obs = torch.rand(1, 3, spec.h, spec.w)
    mask = torch.zeros(1, 1, spec.h, spec.w)
    file_id = torch.tensor([FILE_TO_ID[spec.file_name]], dtype=torch.long)
    with torch.no_grad():
        out = model(obs, mask, file_id)
    assert out.shape == (1, 3, 9, 42)
    assert torch.isfinite(out).all()
