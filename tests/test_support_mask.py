from __future__ import annotations

import json
from pathlib import Path

import torch

from atlas_ai.export_spec import TRAINABLE_EXPORT_SPECS, blank_atlas_like_files
from atlas_ai.support_mask import load_support_masks
from models.losses import exported_files_loss


REPO = Path(__file__).resolve().parents[1]


def test_support_masks_match_export_dimensions():
    masks = load_support_masks()
    spec_by_name = {s.file_name: s for s in TRAINABLE_EXPORT_SPECS}
    for name, mask in masks.items():
        spec = spec_by_name[name]
        assert mask.shape == (spec.h, spec.w), name
        assert mask.dtype == torch.bool, name
        # Cranamp reads at least one pixel from every supported file.
        assert mask.any(), name


def test_pledit_excludes_unrendered_lower_band():
    # PLEDIT rows beyond y=110 carry font/menu sprites Cranamp never uses.
    profile = json.loads((REPO / "configs/supported_pixels_classic.json").read_text())
    pledit_rects = profile["PLEDIT.bmp"]
    for x, y, w, h in pledit_rects:
        assert y + h <= 110, f"PLEDIT rect {(x, y, w, h)} extends into unused band"


def _half_baseline_files():
    # logits=0 -> sigmoid=0.5; pair with target=0.5 for zero-error baseline.
    files = {
        s.file_name: torch.zeros(1, 3, s.h, s.w, requires_grad=True)
        for s in TRAINABLE_EXPORT_SPECS
    }
    target = blank_atlas_like_files({k: v.detach() for k, v in files.items()})
    target = target.fill_(0.5)
    return files, target


def test_loss_ignores_pixels_outside_support_mask():
    files, target = _half_baseline_files()
    target_with_off = target.clone()
    masks = load_support_masks()
    plot_spec = next(s for s in TRAINABLE_EXPORT_SPECS if s.file_name == "PLEDIT.bmp")
    mask = masks["PLEDIT.bmp"]
    off_pixels = (~mask).nonzero(as_tuple=False)
    assert off_pixels.numel(), "PLEDIT should have unsupported pixels"
    py, px = off_pixels[0].tolist()
    target_with_off[..., plot_spec.y + py, plot_spec.x + px] = 1.0

    metrics_a = exported_files_loss(files, target)
    metrics_b = exported_files_loss(files, target_with_off)
    assert torch.isclose(metrics_a["mae_pledit"], metrics_b["mae_pledit"], atol=1e-7)
    assert torch.isclose(metrics_a["total"], metrics_b["total"], atol=1e-7)


def test_loss_is_sensitive_inside_support_mask():
    files, target = _half_baseline_files()
    masks = load_support_masks()

    target_with_on = target.clone()
    plot_spec = next(s for s in TRAINABLE_EXPORT_SPECS if s.file_name == "PLEDIT.bmp")
    mask = masks["PLEDIT.bmp"]
    on_pixels = mask.nonzero(as_tuple=False)
    py, px = on_pixels[0].tolist()
    target_with_on[..., plot_spec.y + py, plot_spec.x + px] = 1.0

    base = exported_files_loss(files, target)
    bumped = exported_files_loss(files, target_with_on)
    assert bumped["mae_pledit"] > base["mae_pledit"]


def test_metrics_invariant_to_batch_size():
    # Same content repeated across a batch must report the same metric as a single sample.
    files1 = {
        s.file_name: torch.full((1, 3, s.h, s.w), 0.4, requires_grad=True)
        for s in TRAINABLE_EXPORT_SPECS
    }
    files2 = {k: v.detach().repeat(2, 1, 1, 1).requires_grad_() for k, v in files1.items()}
    target1 = torch.full((1, 3, 1024, 1024), 0.5)
    target2 = target1.repeat(2, 1, 1, 1)

    m1 = exported_files_loss(files1, target1)
    m2 = exported_files_loss(files2, target2)
    for key in ("mae_pledit", "mae_main", "exported_l1", "total"):
        assert torch.allclose(m1[key], m2[key], atol=1e-7), key


def test_masked_mean_does_not_overflow_large_fp16_masks():
    from models.losses import _masked_mean

    values = torch.ones((1, 6, 315, 275), dtype=torch.float16)
    mask = torch.ones((315, 275), dtype=torch.float16)

    mean = _masked_mean(values, mask)

    assert torch.isfinite(mean)
    assert torch.allclose(mean, torch.tensor(1.0), atol=1e-6)


def test_unsupported_pixels_receive_zero_gradient():
    files, target = _half_baseline_files()
    masks = load_support_masks()

    # Force a non-trivial gradient by perturbing the target inside the supported
    # region of every file. Unsupported pixels must remain at exactly zero grad.
    perturbed = target.clone()
    for spec in TRAINABLE_EXPORT_SPECS:
        mask = masks[spec.file_name]
        on = mask.nonzero(as_tuple=False)
        if not on.numel():
            continue
        py, px = on[0].tolist()
        perturbed[..., spec.y + py, spec.x + px] = 1.0

    loss = exported_files_loss(files, perturbed)["total"]
    loss.backward()
    for spec in TRAINABLE_EXPORT_SPECS:
        grad = files[spec.file_name].grad
        assert grad is not None, spec.file_name
        mask = masks[spec.file_name]
        # Pixels outside the support mask must have exactly zero gradient.
        unsupported = ~mask
        assert torch.all(grad[..., unsupported] == 0), spec.file_name
