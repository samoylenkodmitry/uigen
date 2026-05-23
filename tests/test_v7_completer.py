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
    """With at least some hidden pixels, gradient must reach encoder + output
    head. (Mask=ones disables the generated branch by construction; that case
    is covered separately.)"""
    model = _tiny()
    model.train()
    spec = TRAINABLE_EXPORT_SPECS[0]
    obs = torch.rand(1, 3, spec.h, spec.w)
    mask = torch.zeros(1, 1, spec.h, spec.w)  # all hidden -> full gradient flow
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


# ---------- Skin conditioning (Gate B) ----------

def _tiny_with_skins(num_skins: int = 3) -> V7Completer:
    return V7Completer(
        base_channels=8, file_embedding_dim=8, frequencies=(1, 2),
        num_skins=num_skins, skin_embedding_dim=4,
    )


def test_skin_conditioning_changes_version_buffer_to_71():
    assert int(_tiny_with_skins().model_version.item()) == 71


def test_skin_conditioning_records_buffers():
    m = _tiny_with_skins(num_skins=5)
    assert int(m.num_skins_buffer.item()) == 5
    assert int(m.skin_embedding_dim_buffer.item()) == 4


def test_skin_id_required_when_num_skins_positive():
    model = _tiny_with_skins().eval()
    spec = TRAINABLE_EXPORT_SPECS[0]
    obs = torch.rand(1, 3, spec.h, spec.w)
    mask = torch.zeros(1, 1, spec.h, spec.w)
    file_id = torch.tensor([0], dtype=torch.long)
    with pytest.raises(ValueError, match="skin_id is required"):
        model(obs, mask, file_id)


def test_same_skin_id_deterministic_output():
    torch.manual_seed(0)
    model = _tiny_with_skins().eval()
    spec = TRAINABLE_EXPORT_SPECS[0]
    obs = torch.zeros(1, 3, spec.h, spec.w)
    mask = torch.zeros(1, 1, spec.h, spec.w)
    file_id = torch.tensor([0], dtype=torch.long)
    sid = torch.tensor([1], dtype=torch.long)
    with torch.no_grad():
        a = model(obs, mask, file_id, skin_id=sid)
        b = model(obs, mask, file_id, skin_id=sid)
    assert torch.equal(a, b)


def test_different_skin_id_can_change_output():
    torch.manual_seed(0)
    model = _tiny_with_skins().eval()
    spec = TRAINABLE_EXPORT_SPECS[0]
    # All-hidden input so the generated branch fully drives the output.
    obs = torch.zeros(1, 3, spec.h, spec.w)
    mask = torch.zeros(1, 1, spec.h, spec.w)
    file_id = torch.tensor([0], dtype=torch.long)
    with torch.no_grad():
        out0 = model(obs, mask, file_id, skin_id=torch.tensor([0], dtype=torch.long))
        out1 = model(obs, mask, file_id, skin_id=torch.tensor([1], dtype=torch.long))
    assert not torch.allclose(out0, out1, atol=1e-6), (
        "skin embedding had no measurable effect on a zero-input forward pass"
    )


def test_skin_checkpoint_roundtrip(tmp_path):
    from safetensors.torch import load_file, save_file

    torch.manual_seed(123)
    model = _tiny_with_skins(num_skins=4).eval()
    spec = TRAINABLE_EXPORT_SPECS[6]
    obs = torch.rand(1, 3, spec.h, spec.w)
    mask = torch.ones(1, 1, spec.h, spec.w) * 0.5  # mixed observed
    file_id = torch.tensor([6], dtype=torch.long)
    sid = torch.tensor([2], dtype=torch.long)
    with torch.no_grad():
        out_a = model(obs, mask, file_id, skin_id=sid)
    path = tmp_path / "v7_skin.safetensors"
    save_file({k: v.detach().clone() for k, v in model.state_dict().items()}, str(path))
    loaded = load_file(str(path))
    # Buffers must carry forward.
    assert int(loaded["num_skins_buffer"].item()) == 4
    assert int(loaded["skin_embedding_dim_buffer"].item()) == 4
    assert int(loaded["model_version"].item()) == 71
    fresh = _tiny_with_skins(num_skins=4).eval()
    fresh.load_state_dict(loaded)
    with torch.no_grad():
        out_b = fresh(obs, mask, file_id, skin_id=sid)
    assert torch.allclose(out_a, out_b, atol=1e-6)


def test_no_skin_path_still_works():
    """num_skins=0 keeps the old forward signature working."""
    model = _tiny().eval()
    assert model.skin_embedding is None
    assert int(model.num_skins_buffer.item()) == 0
    assert int(model.skin_embedding_dim_buffer.item()) == 0
    spec = TRAINABLE_EXPORT_SPECS[0]
    obs = torch.rand(1, 3, spec.h, spec.w)
    mask = torch.zeros(1, 1, spec.h, spec.w)
    file_id = torch.tensor([0], dtype=torch.long)
    # No skin_id passed.
    with torch.no_grad():
        out = model(obs, mask, file_id)
    assert out.shape == (1, 3, spec.h, spec.w)


def test_skin_dim_zero_with_num_skins_positive_rejected():
    with pytest.raises(ValueError, match="skin_embedding_dim"):
        V7Completer(num_skins=3, skin_embedding_dim=0)


def test_hard_mask_passthrough_is_exact_before_training():
    """When observed_mask == 1 everywhere, final_rgb must equal observed_rgb
    pixel-for-pixel, without any training, regardless of model parameters."""
    torch.manual_seed(0)
    model = _tiny().eval()
    spec = TRAINABLE_EXPORT_SPECS[0]
    observed_rgb = torch.rand(2, 3, spec.h, spec.w)
    observed_mask = torch.ones(2, 1, spec.h, spec.w)
    file_id = torch.tensor([0, 0], dtype=torch.long)
    with torch.no_grad():
        out = model(observed_rgb, observed_mask, file_id)
    assert torch.equal(out, observed_rgb)


def test_hard_mask_partial_observed_copies_only_observed_pixels():
    """When mask is 1 in some pixels and 0 in others, the final output must
    equal observed_rgb at mask==1 positions, regardless of the generated
    branch's output at those positions."""
    torch.manual_seed(0)
    model = _tiny().eval()
    spec = TRAINABLE_EXPORT_SPECS[0]
    observed_rgb = torch.rand(1, 3, spec.h, spec.w)
    observed_mask = torch.zeros(1, 1, spec.h, spec.w)
    # Reveal a centered rectangle.
    observed_mask[..., 10:50, 30:120] = 1.0
    file_id = torch.tensor([0], dtype=torch.long)
    with torch.no_grad():
        out = model(observed_rgb, observed_mask, file_id)
    mask3 = observed_mask.expand_as(out).bool()
    assert torch.equal(out[mask3], observed_rgb[mask3])


def test_hidden_pixels_come_from_generated_branch():
    """When the input observed_rgb is zeroed (mask=0), the output must NOT be
    zeroed - it comes from sigmoid(rgb_logits) which is in (0, 1)."""
    torch.manual_seed(0)
    model = _tiny().eval()
    spec = TRAINABLE_EXPORT_SPECS[0]
    observed_rgb = torch.zeros(1, 3, spec.h, spec.w)
    observed_mask = torch.zeros(1, 1, spec.h, spec.w)
    file_id = torch.tensor([0], dtype=torch.long)
    with torch.no_grad():
        out = model(observed_rgb, observed_mask, file_id)
    # sigmoid output is strictly in (0, 1); cannot equal the all-zero input.
    assert (out > 0.0).all()
    assert (out < 1.0).all()


def test_support_masked_l1_zero_for_all_observed_within_support_pre_training():
    """The completer's hard-copy contract makes support_masked_l1_loss exactly
    zero when observed_mask is the support mask and observed_rgb is the
    target, regardless of model parameters."""
    from atlas_ai.support_mask import load_support_masks
    from models.losses_v7 import support_masked_l1_loss

    torch.manual_seed(7)
    model = _tiny().eval()
    spec = TRAINABLE_EXPORT_SPECS[6]  # EQMAIN
    file_id = torch.tensor([6], dtype=torch.long)
    target_rgb = torch.rand(1, 3, spec.h, spec.w)
    support = load_support_masks()[spec.file_name].to(torch.float32)  # [H, W]
    mask = support.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    observed_rgb = target_rgb * mask  # zero outside support
    with torch.no_grad():
        out = model(observed_rgb, mask, file_id)
    loss = support_masked_l1_loss(out, target_rgb, support)
    assert float(loss) == 0.0


def test_gradient_flows_to_generated_branch_when_mask_has_hidden():
    """With hidden pixels (mask=0), backprop must reach the generated branch
    (out_proj, fuse layers, encoder). When mask is fully 1, no gradient
    reaches the generated branch by construction."""
    spec = TRAINABLE_EXPORT_SPECS[0]
    # First: mixed mask -> gradient flows.
    model = _tiny()
    model.train()
    observed_rgb = torch.rand(1, 3, spec.h, spec.w)
    observed_mask = torch.zeros(1, 1, spec.h, spec.w)
    observed_mask[..., :spec.h // 2, :] = 1.0  # bottom half hidden
    file_id = torch.tensor([0], dtype=torch.long)
    target = torch.rand(1, 3, spec.h, spec.w)
    out = model(observed_rgb, observed_mask, file_id)
    (out - target).abs().mean().backward()
    assert model.out_proj.weight.grad is not None
    assert model.out_proj.weight.grad.abs().sum() > 0
    assert model.stem[0].weight.grad is not None
    assert model.stem[0].weight.grad.abs().sum() > 0
    # Second: fully observed mask -> no gradient to generated branch.
    model = _tiny()
    model.train()
    full_mask = torch.ones(1, 1, spec.h, spec.w)
    out = model(observed_rgb, full_mask, file_id)
    (out - target).abs().mean().backward()
    # out_proj only contributes through (1 - mask) which is zero everywhere -
    # so its weight gradient is exactly zero.
    assert model.out_proj.weight.grad is None or model.out_proj.weight.grad.abs().sum() == 0


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
