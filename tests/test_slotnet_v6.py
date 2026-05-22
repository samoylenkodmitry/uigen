"""SlotNetV6 model: shape, activation bounds, checkpoint round-trip."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from atlas_ai.export_spec import TRAINABLE_EXPORT_SPECS
from models.slotnet_v6 import SlotNetV6, _decoder_upsample_blocks


def _tiny_model() -> SlotNetV6:
    return SlotNetV6(
        base_channels=8,
        style_dim=32,
        head_channels=16,
        attn_dim=32,
        attention_heads=4,
        cross_attention_layers=1,
        file_embedding_dim=8,
        query_grid_divisor=4,
    )


def test_forward_returns_uv_and_conf_per_file():
    model = _tiny_model().eval()
    view = torch.rand(1, 3, 64, 64)
    out = model(view)
    files = out["files"]
    assert set(files) == {spec.file_name for spec in TRAINABLE_EXPORT_SPECS}
    for spec in TRAINABLE_EXPORT_SPECS:
        entry = files[spec.file_name]
        assert entry["uv"].shape == (1, 2, spec.h, spec.w)
        assert entry["conf_logits"].shape == (1, 1, spec.h, spec.w)


def test_uv_is_in_signed_unit_range():
    model = _tiny_model().eval()
    view = torch.rand(1, 3, 64, 64)
    out = model(view)
    for spec in TRAINABLE_EXPORT_SPECS:
        uv = out["files"][spec.file_name]["uv"]
        assert torch.isfinite(uv).all()
        assert uv.min().item() >= -1.0 - 1e-6
        assert uv.max().item() <= 1.0 + 1e-6


def test_return_attention_exposes_per_file_summaries():
    model = _tiny_model().eval()
    view = torch.rand(1, 3, 64, 64)
    out = model(view, return_attention=True)
    assert set(out["attention"]) == {spec.file_name for spec in TRAINABLE_EXPORT_SPECS}
    for spec in TRAINABLE_EXPORT_SPECS:
        attn = out["attention"][spec.file_name]
        assert attn.dim() == 3
        assert attn.shape[0] == 1  # batch


def test_checkpoint_roundtrip(tmp_path):
    from safetensors.torch import load_file, save_file

    model = _tiny_model().eval()
    view = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        out_a = model(view)
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    path = tmp_path / "v6.safetensors"
    save_file(state, str(path))
    loaded = load_file(str(path))

    fresh = _tiny_model().eval()
    fresh.load_state_dict(loaded)
    with torch.no_grad():
        out_b = fresh(view)
    for spec in TRAINABLE_EXPORT_SPECS:
        for key in ("uv", "conf_logits"):
            assert torch.allclose(out_a["files"][spec.file_name][key],
                                  out_b["files"][spec.file_name][key], atol=1e-6)


def test_version_buffer_is_60():
    model = _tiny_model()
    assert int(model.slotnet_version.item()) == 60


def test_grad_flows_to_encoder_and_every_head():
    model = _tiny_model()
    model.train()
    view = torch.rand(2, 3, 64, 64, requires_grad=False)
    out = model(view)
    loss = torch.zeros((), requires_grad=True)
    for spec in TRAINABLE_EXPORT_SPECS:
        loss = loss + out["files"][spec.file_name]["uv"].abs().mean()
        loss = loss + out["files"][spec.file_name]["conf_logits"].abs().mean()
    loss.backward()
    assert model.enc1[0].weight.grad is not None
    assert model.enc1[0].weight.grad.abs().sum() > 0
    for spec in TRAINABLE_EXPORT_SPECS:
        head_key = spec.file_name.lower().removesuffix(".bmp")
        head = model.heads[head_key]
        assert head.uv_proj.weight.grad is not None
        assert head.conf_proj.weight.grad is not None


def test_decoder_block_count_matches_divisor():
    assert _decoder_upsample_blocks(1) == 0
    assert _decoder_upsample_blocks(2) == 1
    assert _decoder_upsample_blocks(4) == 2
    assert _decoder_upsample_blocks(8) == 3


def test_invalid_attn_dim_raises():
    with pytest.raises(ValueError):
        SlotNetV6(attn_dim=33, attention_heads=4)
    with pytest.raises(ValueError):
        SlotNetV6(attn_dim=10, attention_heads=4)  # not divisible by 4
