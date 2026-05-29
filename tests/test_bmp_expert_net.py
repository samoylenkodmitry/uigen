"""BMPExpertNet: forward [1,3,1728,960] -> exact [1,3,H,W] for every TRAINABLE
BMP size, and round-trip reconstruction from a saved state_dict (buffers carry
the constructor knobs needed to rebuild the model)."""

from __future__ import annotations

import torch

from atlas_ai.dataset_v10_bmp import CANVAS_H, CANVAS_W
from atlas_ai.export_spec import TRAINABLE_EXPORT_SPECS
from models.bmp_expert_net import BMPExpertNet, BMPPatchDiscriminator


def test_patch_discriminator_all_bmp_sizes_and_grad():
    """PatchGAN must forward + backprop for every trainable BMP size (size-aware
    layer cap handles thin/small BMPs) and return a logit map + feature list."""
    for spec in TRAINABLE_EXPORT_SPECS:
        d = BMPPatchDiscriminator(base=16, n_layers=3, min_dim=min(spec.h, spec.w))
        x = torch.rand(2, 3, spec.h, spec.w, requires_grad=True)
        logit, feats = d(x)
        assert logit.dim() == 4 and logit.shape[1] == 1, (spec.file_name, logit.shape)
        assert len(feats) >= 1
        logit.mean().backward()
        assert x.grad is not None


# Use small dims so all 11 sizes fit in CPU memory + time for the test.
_TINY = dict(base=8, attn_dim=32, dec_ch=16, heads=2, attn_layers=1)


def _build(target_h: int, target_w: int) -> BMPExpertNet:
    return BMPExpertNet(target_h=target_h, target_w=target_w, **_TINY)


def test_forward_shape_for_every_trainable_bmp():
    x = torch.rand(1, 3, CANVAS_H, CANVAS_W)
    for spec in TRAINABLE_EXPORT_SPECS:
        m = _build(spec.h, spec.w)
        with torch.no_grad():
            out = m(x)
        assert tuple(out.shape) == (1, 3, spec.h, spec.w), (spec.file_name, out.shape)
        assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_state_dict_round_trip_rebuilds_model():
    # Train-time model with non-default knobs.
    m = BMPExpertNet(target_h=115, target_w=275, base=12, attn_dim=48, dec_ch=24,
                     heads=3, attn_layers=2)
    state = m.state_dict()
    # Buffers must carry every constructor knob needed by inference.
    g = lambda k: int(state[k].reshape(-1)[0].item())
    rebuilt = BMPExpertNet(
        target_h=g("target_h_buf"), target_w=g("target_w_buf"),
        base=g("base_buf"), attn_dim=g("attn_dim_buf"), dec_ch=g("dec_ch_buf"),
        heads=g("heads_buf"), attn_layers=g("attn_layers_buf"),
    )
    rebuilt.load_state_dict(state)
    x = torch.rand(1, 3, CANVAS_H, CANVAS_W)
    with torch.no_grad():
        out = rebuilt(x)
    assert tuple(out.shape) == (1, 3, 115, 275)


def test_input_size_is_flexible():
    # Encoder is fully convolutional; KV pool is adaptive -> shape-agnostic input.
    m = _build(target_h=24, target_w=56)
    for hw in [(CANVAS_H, CANVAS_W), (864, 480), (432, 240)]:
        x = torch.rand(1, 3, *hw)
        with torch.no_grad():
            out = m(x)
        assert tuple(out.shape) == (1, 3, 24, 56)
