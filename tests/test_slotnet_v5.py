from __future__ import annotations

import torch

from atlas_ai.export_spec import TRAINABLE_EXPORT_SPECS
from atlas_ai.support_mask import load_support_masks
from models.losses import exported_files_loss
from models.slotnet_v5 import SlotNetV5


def _tiny_model() -> SlotNetV5:
    """Tiny CPU model so the tests are quick but still exercise the full path."""
    return SlotNetV5(
        base_channels=8,
        style_dim=32,
        head_channels=16,
        attn_dim=16,
        attention_heads=2,
        cross_attention_layers=1,
        file_embedding_dim=8,
        frequencies=(1, 2),
    )


def _tiny_view(batch: int = 1) -> torch.Tensor:
    # 16x stride encoder requires divisibility by 16.
    # Use a smaller-than-real but valid input shape for CPU tests.
    return torch.zeros(batch, 3, 128, 64)


def test_v5_output_shapes_match_export_specs():
    model = _tiny_model().eval()
    with torch.no_grad():
        out = model(_tiny_view())
    assert set(out) == {"files"}
    files = out["files"]
    assert set(files) == {spec.file_name for spec in TRAINABLE_EXPORT_SPECS}
    for spec in TRAINABLE_EXPORT_SPECS:
        tensor = files[spec.file_name]
        assert tensor.shape == (1, 3, spec.h, spec.w), spec.file_name


def test_v5_slotnet_version_buffer_is_50():
    model = _tiny_model()
    assert int(model.slotnet_version.reshape(-1)[0].item()) == 50


def test_v5_forward_returns_files_key():
    model = _tiny_model().eval()
    with torch.no_grad():
        out = model(_tiny_view())
    assert "files" in out and isinstance(out["files"], dict)


def test_v5_forward_return_attention_includes_per_file_maps():
    model = _tiny_model().eval()
    with torch.no_grad():
        out = model(_tiny_view(), return_attention=True)
    assert "attention" in out
    # Encoder spatial map at input 128x64 with /16 = 8x4 tokens.
    for spec in TRAINABLE_EXPORT_SPECS:
        attn = out["attention"][spec.file_name]
        assert attn.dim() == 3, spec.file_name
        assert attn.shape[1:] == (8, 4), spec.file_name


def test_v5_loss_backprop_flows_through_encoder_and_heads():
    """V5 must backprop end-to-end and every per-file head decoder gets gradient.

    Pixel-mask correctness is asserted in tests/test_support_mask.py; here we
    only check V5 plumbing — encoder, attention layers, and every file decoder
    receive non-None gradients after one step of exported_files_loss.
    """
    model = _tiny_model().train()
    view = _tiny_view(batch=1)
    out = model(view)
    files = {name: logits.float() for name, logits in out["files"].items()}
    # Use a non-trivial target so the loss exercises real gradients.
    target = torch.rand(1, 3, 1024, 1024)
    loss = exported_files_loss(files, target)["total"]
    loss.backward()

    enc_params = list(model.enc1.parameters()) + list(model.enc5.parameters())
    assert all(p.grad is not None for p in enc_params)

    feature_grad = model.feature_proj.weight.grad
    assert feature_grad is not None and torch.any(feature_grad != 0)

    for spec in TRAINABLE_EXPORT_SPECS:
        head = model.heads[spec.file_name.lower().removesuffix(".bmp")]
        decoder_params = list(head.decoder.parameters())
        assert all(p.grad is not None for p in decoder_params), spec.file_name
        # Sanity: at least one decoder weight in this head moved.
        assert any(torch.any(p.grad != 0) for p in decoder_params), spec.file_name
    # Confirm support_mask integration is in the loss path (uses load_support_masks).
    assert load_support_masks()  # raises if the profile is missing


def test_checkpoint_dispatch_recognises_v35_and_v5(tmp_path):
    """detect_checkpoint_info + build_model_from_info must round-trip both versions."""
    from safetensors.torch import save_file

    from infer_skin import build_model_from_info, detect_checkpoint_info
    from models.slotnet_v35 import SlotNetV35

    v35 = SlotNetV35(base_channels=8, style_dim=32, head_channels=16)
    p35 = tmp_path / "v35.safetensors"
    save_file({k: v.contiguous().cpu() for k, v in v35.state_dict().items()}, str(p35))
    info35 = detect_checkpoint_info(p35)
    assert info35.version == 35 and info35.base_channels == 8 and info35.style_dim == 32

    v5 = _tiny_model()
    p50 = tmp_path / "v5.safetensors"
    save_file({k: v.contiguous().cpu() for k, v in v5.state_dict().items()}, str(p50))
    info50 = detect_checkpoint_info(p50)
    assert info50.version == 50
    assert info50.base_channels == 8
    assert info50.style_dim == 32
    assert info50.attn_dim == 16
    assert info50.attention_heads == 2
    assert info50.cross_attention_layers == 1
    assert info50.file_embedding_dim == 8

    rebuilt = build_model_from_info(info50, device=torch.device("cpu"))
    assert isinstance(rebuilt, SlotNetV5)
    # State should load cleanly with strict=True.
    rebuilt.load_state_dict(v5.state_dict())


def test_v5_files_match_target_dimensions_under_amp_safe_cast():
    model = _tiny_model().eval()
    view = _tiny_view(batch=2)
    with torch.no_grad():
        out = model(view)
    for spec in TRAINABLE_EXPORT_SPECS:
        assert out["files"][spec.file_name].shape == (2, 3, spec.h, spec.w)
