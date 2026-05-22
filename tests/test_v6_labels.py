"""V6 label generation from final-frame provenance."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
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


def _make_renderer(cm, canvas_w: int = 64, canvas_h: int = 64):
    return cm.Renderer(ROOT / "assets/default_skin", canvas_w, canvas_h)


def _inject(renderer, cm, file_name: str, rgba: np.ndarray) -> None:
    from atlas_ai.skins import normalize_name
    renderer.images[normalize_name(file_name)] = Image.fromarray(rgba, mode="RGBA")


def _solid_rgba(h: int, w: int, *, color=(255, 0, 0), alpha: int = 255) -> np.ndarray:
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[..., 0] = color[0]
    arr[..., 1] = color[1]
    arr[..., 2] = color[2]
    arr[..., 3] = alpha
    return arr


def _rand_rgba(h: int, w: int, *, seed: int = 0, alpha: int = 255) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[..., :3] = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    arr[..., 3] = alpha
    return arr


def test_label_shapes_and_dtypes():
    from atlas_ai.v6_labels import build_v6_labels
    from atlas_ai.export_spec import TRAINABLE_EXPORT_SPECS

    canvas_w, canvas_h = 64, 32
    provenance = np.zeros((canvas_h, canvas_w), dtype=np.uint32)
    labels = build_v6_labels(provenance, canvas_w, canvas_h)
    assert set(labels) == {spec.file_name for spec in TRAINABLE_EXPORT_SPECS}
    for spec in TRAINABLE_EXPORT_SPECS:
        entry = labels[spec.file_name]
        assert entry["visible_mask"].shape == (spec.h, spec.w)
        assert entry["visible_mask"].dtype == np.uint8
        assert entry["uv_target"].shape == (2, spec.h, spec.w)
        assert entry["uv_target"].dtype == np.float32


def test_empty_provenance_yields_empty_masks():
    from atlas_ai.v6_labels import build_v6_labels

    canvas_w, canvas_h = 64, 32
    provenance = np.zeros((canvas_h, canvas_w), dtype=np.uint32)
    labels = build_v6_labels(provenance, canvas_w, canvas_h)
    for entry in labels.values():
        assert entry["visible_mask"].sum() == 0
        assert np.abs(entry["uv_target"]).sum() == 0


def test_shape_mismatch_raises():
    from atlas_ai.v6_labels import build_v6_labels

    bad = np.zeros((16, 16), dtype=np.uint32)
    with pytest.raises(ValueError):
        build_v6_labels(bad, canvas_w=32, canvas_h=16)


def test_dtype_mismatch_raises():
    from atlas_ai.v6_labels import build_v6_labels

    bad = np.zeros((16, 16), dtype=np.int32)
    with pytest.raises(TypeError):
        build_v6_labels(bad, canvas_w=16, canvas_h=16)


def test_unknown_file_id_raises(cm):
    from atlas_ai.v6_labels import build_v6_labels

    canvas_w, canvas_h = 8, 8
    provenance = np.zeros((canvas_h, canvas_w), dtype=np.uint32)
    provenance[0, 0] = cm.encode_provenance(99, 0, 0)
    with pytest.raises(ValueError, match="unknown trainable file id"):
        build_v6_labels(provenance, canvas_w, canvas_h)


def test_out_of_bounds_source_coordinate_raises(cm):
    from atlas_ai.v6_labels import build_v6_labels

    canvas_w, canvas_h = 8, 8
    provenance = np.zeros((canvas_h, canvas_w), dtype=np.uint32)
    main_id = cm.TRAINABLE_FILE_TO_ID["MAIN.bmp"]
    provenance[0, 0] = cm.encode_provenance(main_id, 0, 999)
    with pytest.raises(ValueError, match="out-of-bounds source coordinate"):
        build_v6_labels(provenance, canvas_w, canvas_h)


def test_sprite_identity_label(cm):
    """Single 1:1 blit must produce a visible mask and UV pointing at the blit location."""
    from atlas_ai.v6_labels import build_v6_labels

    canvas_w, canvas_h = 64, 64
    r = _make_renderer(cm, canvas_w, canvas_h)
    _inject(r, cm, "MAIN.bmp", _solid_rgba(4, 4))
    r.blit("MAIN", "MAIN.bmp", (0, 0, 4, 4), (10, 20), 1.0)
    labels = build_v6_labels(r.provenance, canvas_w, canvas_h)
    main = labels["MAIN.bmp"]

    # Source pixels (0..3, 0..3) are visible, the rest are not.
    assert main["visible_mask"][:4, :4].all()
    assert main["visible_mask"].sum() == 16

    # UV at (sy=0, sx=0) points to canvas (10, 20).
    u = float(main["uv_target"][0, 0, 0])
    v = float(main["uv_target"][1, 0, 0])
    expected_u = 2.0 * (10 + 0.5) / canvas_w - 1.0
    expected_v = 2.0 * (20 + 0.5) / canvas_h - 1.0
    assert abs(u - expected_u) < 5e-4
    assert abs(v - expected_v) < 5e-4

    # UV at (sy=3, sx=3) points to canvas (13, 23).
    u = float(main["uv_target"][0, 3, 3])
    v = float(main["uv_target"][1, 3, 3])
    expected_u = 2.0 * (13 + 0.5) / canvas_w - 1.0
    expected_v = 2.0 * (23 + 0.5) / canvas_h - 1.0
    assert abs(u - expected_u) < 5e-4
    assert abs(v - expected_v) < 5e-4


def test_scaled_blit_first_canvas_pixel_wins(cm):
    """At scale 2.0 the top-left of a source pixel's 2x2 block must be the chosen UV."""
    from atlas_ai.v6_labels import build_v6_labels

    canvas_w, canvas_h = 32, 32
    r = _make_renderer(cm, canvas_w, canvas_h)
    _inject(r, cm, "MAIN.bmp", _solid_rgba(4, 4))
    r.blit("MAIN", "MAIN.bmp", (0, 0, 4, 4), (0, 0), 2.0)
    labels = build_v6_labels(r.provenance, canvas_w, canvas_h)
    main = labels["MAIN.bmp"]

    # Source (0, 0) is rendered into a 2x2 block at canvas (0..1, 0..1).
    # The first canvas pixel (row-major) is (0, 0).
    u = float(main["uv_target"][0, 0, 0])
    v = float(main["uv_target"][1, 0, 0])
    expected_u = 2.0 * (0 + 0.5) / canvas_w - 1.0
    expected_v = 2.0 * (0 + 0.5) / canvas_h - 1.0
    assert abs(u - expected_u) < 5e-4
    assert abs(v - expected_v) < 5e-4
    # Source (1, 1) is rendered into canvas (2..3, 2..3); first pixel is (2, 2).
    u11 = float(main["uv_target"][0, 1, 1])
    v11 = float(main["uv_target"][1, 1, 1])
    expected_u11 = 2.0 * (2 + 0.5) / canvas_w - 1.0
    expected_v11 = 2.0 * (2 + 0.5) / canvas_h - 1.0
    assert abs(u11 - expected_u11) < 5e-4
    assert abs(v11 - expected_v11) < 5e-4


def test_save_load_roundtrip(cm, tmp_path):
    from atlas_ai.v6_labels import build_v6_labels, save_v6_labels, load_v6_labels

    canvas_w, canvas_h = 64, 64
    r = _make_renderer(cm, canvas_w, canvas_h)
    _inject(r, cm, "MAIN.bmp", _solid_rgba(4, 4))
    _inject(r, cm, "CBUTTONS.bmp", _solid_rgba(3, 3))
    r.blit("MAIN", "MAIN.bmp", (0, 0, 4, 4), (10, 10), 1.0)
    r.blit("CBUTTONS", "CBUTTONS.bmp", (0, 0, 3, 3), (30, 30), 1.0)
    labels = build_v6_labels(r.provenance, canvas_w, canvas_h)

    out = tmp_path / "labels.npz"
    save_v6_labels(labels, out)
    assert out.exists()
    reloaded = load_v6_labels(out)
    assert set(reloaded) == set(labels)
    for file_name in labels:
        assert np.array_equal(
            reloaded[file_name]["visible_mask"], labels[file_name]["visible_mask"]
        )
        assert np.array_equal(
            reloaded[file_name]["uv_target"], labels[file_name]["uv_target"]
        )


def test_label_summary(cm):
    from atlas_ai.v6_labels import build_v6_labels, labels_summary

    canvas_w, canvas_h = 64, 64
    r = _make_renderer(cm, canvas_w, canvas_h)
    _inject(r, cm, "MAIN.bmp", _solid_rgba(4, 4))
    r.blit("MAIN", "MAIN.bmp", (0, 0, 4, 4), (0, 0), 1.0)
    labels = build_v6_labels(r.provenance, canvas_w, canvas_h)
    summary = labels_summary(labels)
    assert summary["MAIN.bmp"]["visible_count"] == 16
    # Other files have zero visible.
    assert summary["EQMAIN.bmp"]["visible_count"] == 0


def _grid_sample_oracle(
    view_rgb_uint8: np.ndarray,
    uv_target: np.ndarray,
    visible_mask: np.ndarray,
    *,
    mode: str = "bilinear",
) -> np.ndarray:
    """Run the V6 oracle copy path on one file.

    Returns a (h, w, 3) uint8 array of grid_sample(view, uv_target) at the
    visible source pixels (zeros elsewhere).
    """
    view = torch.from_numpy(view_rgb_uint8.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    uv = torch.from_numpy(uv_target.astype(np.float32)).unsqueeze(0)  # [1, 2, H, W]
    sampled = F.grid_sample(
        view,
        uv.permute(0, 2, 3, 1),
        mode=mode,
        align_corners=False,
    )  # [1, 3, H, W]
    sampled_rgb = (sampled[0].clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    sampled_rgb[~visible_mask.astype(bool)] = 0
    return sampled_rgb


def test_oracle_grid_sample_reproduces_visible_pixels(cm):
    """Stage 1 sanity: grid_sample(view, uv_target) at visible pixels must match
    the rendered canvas pixel (which equals the clean BMP source for an
    unoccluded sprite blit). Validates the full provenance -> label -> copy chain."""
    from atlas_ai.v6_labels import build_v6_labels
    from atlas_ai.export_spec import TRAINABLE_EXPORT_SPECS

    canvas_w, canvas_h = 64, 64
    r = _make_renderer(cm, canvas_w, canvas_h)
    main_src = _rand_rgba(8, 8, seed=11)
    cb_src = _rand_rgba(6, 6, seed=23)
    _inject(r, cm, "MAIN.bmp", main_src)
    _inject(r, cm, "CBUTTONS.bmp", cb_src)
    r.blit("MAIN", "MAIN.bmp", (0, 0, 8, 8), (20, 30), 1.0)
    r.blit("CBUTTONS", "CBUTTONS.bmp", (0, 0, 6, 6), (50, 10), 1.0)

    canvas_rgb = np.asarray(r.canvas.convert("RGB"), dtype=np.uint8)
    labels = build_v6_labels(r.provenance, canvas_w, canvas_h)

    spec_by_name = {spec.file_name: spec for spec in TRAINABLE_EXPORT_SPECS}

    for file_name, src_arr in [("MAIN.bmp", main_src), ("CBUTTONS.bmp", cb_src)]:
        spec = spec_by_name[file_name]
        visible = labels[file_name]["visible_mask"]
        uv = labels[file_name]["uv_target"]
        assert visible.sum() == src_arr.shape[0] * src_arr.shape[1]
        sampled = _grid_sample_oracle(canvas_rgb, uv, visible)
        # The clean source equals the original sprite RGB inside its (H, W) region,
        # but the V6 visible-mask tensor is sized (spec.h, spec.w). Compare only
        # within the actual sprite footprint.
        sh, sw = src_arr.shape[:2]
        expected = src_arr[:sh, :sw, :3]
        delta = np.abs(sampled[:sh, :sw, :].astype(np.int16) - expected.astype(np.int16))
        # Bilinear grid_sample at an exact pixel center returns that pixel.
        assert delta.max() <= 1, (
            f"{file_name} oracle MAE failure: max delta {delta.max()}"
        )


def test_oracle_works_after_overdraw(cm):
    """Overdraw must invalidate prior labels: the orphaned MAIN pixels must not
    be marked visible, and surviving MAIN pixels still grid_sample correctly."""
    from atlas_ai.v6_labels import build_v6_labels

    canvas_w, canvas_h = 64, 64
    r = _make_renderer(cm, canvas_w, canvas_h)
    main_src = _rand_rgba(8, 8, seed=7)
    cb_src = _rand_rgba(4, 4, seed=8)
    _inject(r, cm, "MAIN.bmp", main_src)
    _inject(r, cm, "CBUTTONS.bmp", cb_src)
    r.blit("MAIN", "MAIN.bmp", (0, 0, 8, 8), (10, 10), 1.0)
    r.blit("CBUTTONS", "CBUTTONS.bmp", (0, 0, 4, 4), (12, 12), 1.0)
    canvas_rgb = np.asarray(r.canvas.convert("RGB"), dtype=np.uint8)
    labels = build_v6_labels(r.provenance, canvas_w, canvas_h)

    main_visible = labels["MAIN.bmp"]["visible_mask"]
    # MAIN source pixels (2..5, 2..5) are now covered by CBUTTONS overdraw.
    assert main_visible[2:6, 2:6].sum() == 0
    # Pixels outside the overdraw region remain visible.
    assert main_visible[0, 0] == 1
    assert main_visible[7, 7] == 1

    sampled = _grid_sample_oracle(canvas_rgb, labels["MAIN.bmp"]["uv_target"], main_visible)
    delta = np.abs(sampled[:8, :8, :].astype(np.int16) - main_src[:8, :8, :3].astype(np.int16))
    # Only check visible region; covered region is zeroed by the oracle helper.
    visible_mask3 = np.repeat(main_visible[:8, :8, None], 3, axis=2).astype(bool)
    assert delta[visible_mask3].max() <= 2


def test_oracle_works_under_window_stretch(cm):
    """Stretched window provenance must reproduce visible pixels after grid_sample."""
    from atlas_ai.v6_labels import build_v6_labels

    canvas_w, canvas_h = 64, 64
    r = _make_renderer(cm, canvas_w, canvas_h)
    main_src = _rand_rgba(4, 4, seed=13)
    _inject(r, cm, "MAIN.bmp", main_src)

    def render_fn(rr, _params):
        rr.blit("MAIN", "MAIN.bmp", (0, 0, 4, 4), (0, 0), 1.0)

    cm._render_window_pass(r, render_fn, {"windows": {"w": [0, 0]}}, "w", (0, 0), 2.0, canvas_w, canvas_h)
    canvas_rgb = np.asarray(r.canvas.convert("RGB"), dtype=np.uint8)
    labels = build_v6_labels(r.provenance, canvas_w, canvas_h)
    visible = labels["MAIN.bmp"]["visible_mask"]
    # All 4x4 source pixels survive (each represented by the first canvas
    # pixel in its stretched block).
    assert visible[:4, :4].sum() == 16

    sampled = _grid_sample_oracle(canvas_rgb, labels["MAIN.bmp"]["uv_target"], visible)
    delta = np.abs(sampled[:4, :4, :].astype(np.int16) - main_src[:4, :4, :3].astype(np.int16))
    assert delta.max() <= 2


def test_oracle_on_full_pipeline_default_skin_nearest(cm):
    """End-to-end label semantics: with nearest-mode grid_sample (which directly
    indexes the canvas pixel at UV), the oracle copy must reproduce the clean
    BMP source at every visible pixel. This validates that the label correctly
    identifies *which* canvas pixel holds each source pixel's value."""
    from atlas_ai.v6_labels import build_v6_labels
    from atlas_ai.export_spec import TRAINABLE_EXPORT_SPECS

    canvas_w, canvas_h = 480, 864
    params = cm.rand_params(seed=7, canvas_w=canvas_w, canvas_h=canvas_h)
    renderer = cm.render_with_params(ROOT / "assets/default_skin", params, canvas_w, canvas_h)
    canvas_rgb = np.asarray(renderer.canvas.convert("RGB"), dtype=np.uint8)
    labels = build_v6_labels(renderer.provenance, canvas_w, canvas_h)

    spec_by_name = {spec.file_name: spec for spec in TRAINABLE_EXPORT_SPECS}
    bmp_cache: dict[str, np.ndarray] = {}
    default_skin = ROOT / "assets/default_skin"

    total_visible = 0
    total_exact = 0
    for file_name, entry in labels.items():
        visible = entry["visible_mask"]
        if visible.sum() == 0:
            continue
        spec = spec_by_name[file_name]
        if file_name not in bmp_cache:
            with Image.open(default_skin / file_name) as im:
                bmp_cache[file_name] = np.asarray(im.convert("RGB"), dtype=np.uint8)
        clean = bmp_cache[file_name]
        assert clean.shape[:2] == (spec.h, spec.w), (
            f"{file_name} BMP shape {clean.shape[:2]} != spec ({spec.h}, {spec.w})"
        )
        sampled = _grid_sample_oracle(canvas_rgb, entry["uv_target"], visible, mode="nearest")
        vmask3 = np.repeat(visible[:, :, None], 3, axis=2).astype(bool)
        delta = np.abs(sampled.astype(np.int16) - clean.astype(np.int16))
        # Nearest mode at float32 UV pixel centers should be exact, with one
        # level tolerated for float math round-trip noise.
        exact = (delta[vmask3] <= 1).sum()
        total_visible += int(vmask3.sum())
        total_exact += int(exact)
    assert total_visible > 0
    # At least 99.5% of visible pixels must round-trip within 1 level under
    # nearest-mode + float32 UV.
    assert total_exact / total_visible > 0.995, (
        f"nearest-mode label round-trip: only {total_exact}/{total_visible} "
        f"({total_exact/total_visible*100:.2f}%) within 1 level"
    )


def test_oracle_on_full_pipeline_default_skin_bilinear_mae(cm):
    """Stage 1 oracle (bilinear, V6 production mode): visible MAE must be near
    zero when labels point at exact input pixel centers."""
    from atlas_ai.v6_labels import build_v6_labels
    from atlas_ai.export_spec import TRAINABLE_EXPORT_SPECS

    canvas_w, canvas_h = 480, 864
    params = cm.rand_params(seed=7, canvas_w=canvas_w, canvas_h=canvas_h)
    renderer = cm.render_with_params(ROOT / "assets/default_skin", params, canvas_w, canvas_h)
    canvas_rgb = np.asarray(renderer.canvas.convert("RGB"), dtype=np.uint8)
    labels = build_v6_labels(renderer.provenance, canvas_w, canvas_h)

    spec_by_name = {spec.file_name: spec for spec in TRAINABLE_EXPORT_SPECS}
    bmp_cache: dict[str, np.ndarray] = {}
    default_skin = ROOT / "assets/default_skin"

    total_abs_delta = 0.0
    total_channels = 0
    for file_name, entry in labels.items():
        visible = entry["visible_mask"]
        if visible.sum() == 0:
            continue
        spec = spec_by_name[file_name]
        if file_name not in bmp_cache:
            with Image.open(default_skin / file_name) as im:
                bmp_cache[file_name] = np.asarray(im.convert("RGB"), dtype=np.uint8)
        clean = bmp_cache[file_name]
        assert clean.shape[:2] == (spec.h, spec.w)
        sampled = _grid_sample_oracle(canvas_rgb, entry["uv_target"], visible, mode="bilinear")
        vmask3 = np.repeat(visible[:, :, None], 3, axis=2).astype(bool)
        delta = np.abs(sampled.astype(np.float32) - clean.astype(np.float32))[vmask3]
        total_abs_delta += float(delta.sum())
        total_channels += int(delta.size)
    assert total_channels > 0
    mae_0_to_255 = total_abs_delta / total_channels
    mae_0_to_1 = mae_0_to_255 / 255.0
    assert mae_0_to_1 < 0.0001, f"bilinear oracle visible MAE = {mae_0_to_1:.6f}"
