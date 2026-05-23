"""V6 loss tests: each term + composite + oracle UV gives near-zero copy_rgb."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
CLI_MODULE_PATH = ROOT / "cranamp_cli/cranamp/tools/cranamp_cli.py"


@pytest.fixture(scope="module")
def cm():
    spec = importlib.util.spec_from_file_location("cranamp_cli_tool", CLI_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_grid_sample_copy_shapes():
    from models.losses_v6 import grid_sample_copy

    view = torch.rand(2, 3, 64, 96)
    uv = torch.zeros(2, 2, 16, 24)
    out = grid_sample_copy(view, uv)
    assert out.shape == (2, 3, 16, 24)


def test_grid_sample_copy_rejects_wrong_shape():
    from models.losses_v6 import grid_sample_copy

    with pytest.raises(ValueError):
        grid_sample_copy(torch.rand(2, 4, 8, 8), torch.zeros(2, 2, 4, 4))
    with pytest.raises(ValueError):
        grid_sample_copy(torch.rand(2, 3, 8, 8), torch.zeros(2, 3, 4, 4))


def test_masked_mean_handles_empty_mask():
    from models.losses_v6 import masked_mean

    x = torch.rand(8, 8)
    mask = torch.zeros(8, 8)
    out = masked_mean(x, mask)
    assert torch.isfinite(out)
    assert float(out) == 0.0


def test_copy_rgb_l1_zero_when_match():
    from models.losses_v6 import copy_rgb_l1_loss

    target = torch.full((1, 3, 4, 4), 0.5)
    pred = target.clone()
    visible = torch.ones(1, 1, 4, 4)
    assert float(copy_rgb_l1_loss(pred, target, visible)) == 0.0


def test_copy_rgb_l1_only_counts_visible():
    from models.losses_v6 import copy_rgb_l1_loss

    target = torch.zeros(1, 3, 4, 4)
    pred = torch.zeros(1, 3, 4, 4)
    pred[..., 0, 0] = 1.0  # large error at (0, 0) only
    visible = torch.zeros(1, 1, 4, 4)
    # No visible pixels include the error; loss must be zero.
    loss = copy_rgb_l1_loss(pred, target, visible)
    assert float(loss) == 0.0
    visible[..., 0, 0] = 1.0
    loss_now = copy_rgb_l1_loss(pred, target, visible)
    # 3 channels with diff 1.0 each, mean over channels = 1.0.
    assert pytest.approx(float(loss_now), abs=1e-6) == 1.0


def test_conf_bce_positive_only_label_finite():
    from models.losses_v6 import conf_bce_loss

    logits = torch.zeros(1, 1, 4, 4)
    visible = torch.ones(1, 1, 4, 4)
    out = conf_bce_loss(logits, visible)
    # With pos_weight cap 10 and all positives, BCE = -log(sigmoid(0)) * 10 = ln(2)*10.
    assert torch.isfinite(out)


def test_conf_bce_balanced_classes_match_plain_bce():
    from models.losses_v6 import conf_bce_loss

    torch.manual_seed(0)
    logits = torch.randn(1, 1, 8, 8)
    visible = torch.zeros(1, 1, 8, 8)
    # 50/50 positives = balanced; pos_weight clamps to 1.
    visible.view(-1)[:32] = 1.0
    out = conf_bce_loss(logits, visible)
    plain = torch.nn.functional.binary_cross_entropy_with_logits(logits, visible)
    assert pytest.approx(float(out), abs=1e-6) == float(plain)


def test_conf_bce_pos_weight_cap():
    """When positives are sparse, pos_weight is clamped to the cap."""
    from models.losses_v6 import DEFAULT_CONF_POS_WEIGHT_CAP, conf_bce_loss

    logits = torch.zeros(1, 1, 32, 32)
    visible = torch.zeros(1, 1, 32, 32)
    visible.view(-1)[0] = 1.0  # one positive in 1024 pixels
    # Without cap, pos_weight would be 1023; with cap it must be 10.
    out = conf_bce_loss(logits, visible, pos_weight_cap=DEFAULT_CONF_POS_WEIGHT_CAP)
    plain_pos = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, visible, pos_weight=torch.tensor(DEFAULT_CONF_POS_WEIGHT_CAP)
    )
    assert pytest.approx(float(out), abs=1e-6) == float(plain_pos)


def test_uv_smooth_l1_only_counts_visible():
    from models.losses_v6 import uv_smooth_l1_loss

    pred = torch.zeros(1, 2, 4, 4)
    target = torch.zeros(1, 2, 4, 4)
    # Set u channel only at pixel (0, 0) to 0.5; v stays at 0.
    target[:, 0, 0, 0] = 0.5
    visible = torch.zeros(1, 1, 4, 4)
    assert float(uv_smooth_l1_loss(pred, target, visible)) == 0.0
    visible[..., 0, 0] = 1.0
    out = uv_smooth_l1_loss(pred, target, visible, beta=0.1)
    # Smooth-L1 with beta 0.1 at |error| 0.5: linear region -> error - 0.5*beta = 0.45.
    # u contributes 0.45, v contributes 0. Mask covers 2 entries (u and v at (0,0)).
    assert pytest.approx(float(out), abs=1e-6) == (0.45 + 0.0) / 2


def test_uv_smooth_l1_pixel_loss_zero_when_match():
    from models.losses_v6 import uv_smooth_l1_pixel_loss

    pred = torch.zeros(1, 2, 4, 4)
    target = torch.zeros(1, 2, 4, 4)
    visible = torch.ones(1, 1, 4, 4)
    out = uv_smooth_l1_pixel_loss(pred, target, visible, view_h=1728, view_w=960, beta_px=2.0)
    assert float(out) == 0.0


def test_uv_smooth_l1_pixel_loss_matches_pixel_math():
    """1 normalized unit = (view_w / 2, view_h / 2) pixels. Verify the
    smooth-L1 evaluation happens in pixel space against beta_px."""
    from models.losses_v6 import uv_smooth_l1_pixel_loss

    view_w, view_h = 100, 200
    pred = torch.zeros(1, 2, 2, 2)
    target = torch.zeros(1, 2, 2, 2)
    # u error 0.1 at (0,0) -> 0.1 * 100/2 = 5 px. v error 0 there.
    target[:, 0, 0, 0] = 0.1
    visible = torch.zeros(1, 1, 2, 2)
    visible[..., 0, 0] = 1.0
    # smooth_l1(5px, beta=2.0): linear region -> 5 - 0.5*2 = 4.0. v contributes 0.
    # Mask covers u and v at (0,0), so mean = (4 + 0) / 2 = 2.0.
    out = uv_smooth_l1_pixel_loss(pred, target, visible, view_h=view_h, view_w=view_w, beta_px=2.0)
    assert pytest.approx(float(out), abs=1e-5) == 2.0


def test_compute_v6_copy_stage_loss_pixel_mode_keys():
    from models.losses_v6 import V6CopyLossWeights, compute_v6_copy_stage_loss

    view = torch.rand(1, 3, 32, 48)
    uv = torch.zeros(1, 2, 8, 8, requires_grad=True)
    conf = torch.zeros(1, 1, 8, 8, requires_grad=True)
    target = torch.rand(1, 3, 8, 8)
    visible = torch.zeros(1, 1, 8, 8)
    visible[..., 2:6, 2:6] = 1.0
    uv_target = torch.zeros(1, 2, 8, 8)
    out = compute_v6_copy_stage_loss(
        view=view,
        uv_pred=uv,
        conf_logits=conf,
        target_rgb=target,
        visible_mask=visible,
        uv_target=uv_target,
        weights=V6CopyLossWeights(),
        uv_loss_mode="pixel",
        uv_beta_px=2.0,
    )
    assert set(out) == {"total", "copy_rgb", "conf", "uv", "uv_tv"}
    out["total"].backward()
    assert uv.grad is not None and uv.grad.abs().sum() > 0


def test_compute_v6_copy_stage_loss_rejects_unknown_mode():
    from models.losses_v6 import V6CopyLossWeights, compute_v6_copy_stage_loss

    view = torch.rand(1, 3, 32, 48)
    uv = torch.zeros(1, 2, 8, 8)
    conf = torch.zeros(1, 1, 8, 8)
    target = torch.zeros(1, 3, 8, 8)
    visible = torch.zeros(1, 1, 8, 8)
    uv_target = torch.zeros(1, 2, 8, 8)
    with pytest.raises(ValueError, match="unknown uv_loss_mode"):
        compute_v6_copy_stage_loss(
            view=view,
            uv_pred=uv,
            conf_logits=conf,
            target_rgb=target,
            visible_mask=visible,
            uv_target=uv_target,
            weights=V6CopyLossWeights(),
            uv_loss_mode="bogus",
        )


def test_uv_tv_zero_on_constant_field():
    from models.losses_v6 import uv_total_variation_loss

    uv = torch.full((1, 2, 4, 4), 0.5)
    assert float(uv_total_variation_loss(uv)) == 0.0


def test_uv_tv_positive_on_noisy_field():
    from models.losses_v6 import uv_total_variation_loss

    torch.manual_seed(0)
    uv = torch.randn(1, 2, 8, 8)
    assert float(uv_total_variation_loss(uv)) > 0.0


def test_compute_v6_copy_stage_loss_keys_and_grad():
    from models.losses_v6 import V6CopyLossWeights, compute_v6_copy_stage_loss

    view = torch.rand(1, 3, 64, 64, requires_grad=True)
    uv = torch.zeros(1, 2, 8, 8, requires_grad=True)
    conf = torch.zeros(1, 1, 8, 8, requires_grad=True)
    target = torch.rand(1, 3, 8, 8)
    visible = torch.zeros(1, 1, 8, 8)
    visible[..., 2:6, 2:6] = 1.0
    uv_target = torch.zeros(1, 2, 8, 8)
    out = compute_v6_copy_stage_loss(
        view=view,
        uv_pred=uv,
        conf_logits=conf,
        target_rgb=target,
        visible_mask=visible,
        uv_target=uv_target,
        weights=V6CopyLossWeights(),
    )
    assert set(out) == {"total", "copy_rgb", "conf", "uv", "uv_tv"}
    out["total"].backward()
    assert uv.grad is not None and uv.grad.abs().sum() > 0
    assert conf.grad is not None and conf.grad.abs().sum() > 0


def _inject(renderer, cm, file_name: str, rgba: np.ndarray) -> None:
    from atlas_ai.skins import normalize_name
    renderer.images[normalize_name(file_name)] = Image.fromarray(rgba, mode="RGBA")


def _rand_rgba(h: int, w: int, *, seed: int = 0, alpha: int = 255) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[..., :3] = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    arr[..., 3] = alpha
    return arr


def test_oracle_uv_gives_low_copy_rgb_loss(cm):
    """Stage 1 sanity wired through the loss: with ground-truth UV the bilinear
    copy loss against the *rendered canvas* must be effectively zero on
    unoccluded blits."""
    from atlas_ai.v6_labels import build_v6_labels
    from atlas_ai.export_spec import TRAINABLE_EXPORT_SPECS
    from models.losses_v6 import copy_rgb_l1_loss, grid_sample_copy

    canvas_w, canvas_h = 64, 64
    renderer = cm.Renderer(ROOT / "assets/default_skin", canvas_w, canvas_h)
    main_src = _rand_rgba(8, 8, seed=3)
    cb_src = _rand_rgba(6, 6, seed=5)
    _inject(renderer, cm, "MAIN.bmp", main_src)
    _inject(renderer, cm, "CBUTTONS.bmp", cb_src)
    renderer.blit("MAIN", "MAIN.bmp", (0, 0, 8, 8), (20, 30), 1.0)
    renderer.blit("CBUTTONS", "CBUTTONS.bmp", (0, 0, 6, 6), (50, 10), 1.0)

    canvas_rgb = np.asarray(renderer.canvas.convert("RGB"), dtype=np.float32) / 255.0
    view = torch.from_numpy(canvas_rgb.transpose(2, 0, 1)).unsqueeze(0)
    labels = build_v6_labels(renderer.provenance, canvas_w, canvas_h)

    spec_by_name = {spec.file_name: spec for spec in TRAINABLE_EXPORT_SPECS}
    for file_name, src in [("MAIN.bmp", main_src), ("CBUTTONS.bmp", cb_src)]:
        spec = spec_by_name[file_name]
        # Labels are sized to the exported BMP spec (e.g. MAIN is 115x275).
        # Place the synthetic sprite at the top-left of a spec-sized target so
        # the visible mask region overlays the same pixels in both tensors.
        uv_target = torch.from_numpy(labels[file_name]["uv_target"]).unsqueeze(0)
        visible = torch.from_numpy(labels[file_name]["visible_mask"]).unsqueeze(0).unsqueeze(0).float()
        target = torch.zeros(1, 3, spec.h, spec.w)
        sh, sw = src.shape[:2]
        src_rgb = torch.from_numpy(src[:sh, :sw, :3].astype(np.float32) / 255.0).permute(2, 0, 1)
        target[0, :, :sh, :sw] = src_rgb
        copy_rgb = grid_sample_copy(view, uv_target).clamp(0.0, 1.0)
        loss = copy_rgb_l1_loss(copy_rgb, target, visible)
        # Bilinear at exact pixel centers under float32 UV: effectively zero
        # for unoccluded interior pixels of the synthetic sprite.
        assert float(loss) < 5e-3, f"{file_name}: oracle copy_rgb loss too high: {float(loss):.4f}"
