"""Renderer provenance correctness tests (per GPT/Codex review).

These verify that `visible_atlas_mask` reflects TRUE final-frame visibility --
i.e., atlas pixels that survived overdraw / fill / erase / transparent /
scaling -- not just "atlas pixels passed to Renderer.blit at some point".
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
from PIL import Image

# Make the cranamp_cli runtime importable as a flat module.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "cranamp_cli" / "cranamp" / "tools"))
import cranamp_cli as cli
from atlas_ai.profiles import Slot


@pytest.fixture
def renderer():
    r = cli.Renderer(skin_source=ROOT / "assets" / "default_skin", canvas_w=64, canvas_h=64)
    # Inject a tiny in-memory atlas with two slots A, B for unit-testing only.
    r.slots = {
        "A": Slot(id=0, name="A", file="A.bmp", x=0,   y=0,   w=16, h=16, loss_weight=1.0),
        "B": Slot(id=1, name="B", file="B.bmp", x=16,  y=0,   w=16, h=16, loss_weight=1.0),
    }
    return r


def _opaque_img(w: int, h: int, rgb=(255, 0, 0)) -> Image.Image:
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[..., 0] = rgb[0]; arr[..., 1] = rgb[1]; arr[..., 2] = rgb[2]
    arr[..., 3] = 255
    return Image.fromarray(arr, "RGBA")


def _half_transparent(w: int, h: int) -> Image.Image:
    """Right half transparent, left half opaque green."""
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[:, : w // 2, 1] = 255  # green
    arr[:, : w // 2, 3] = 255  # opaque left
    arr[:, w // 2 :, 3] = 0    # transparent right
    return Image.fromarray(arr, "RGBA")


def _install_image(renderer, file_name: str, img: Image.Image) -> None:
    """Bypass the on-disk skin loader for tests."""
    from atlas_ai.skins import normalize_name
    renderer.images[normalize_name(file_name)] = img


def test_blit_records_provenance_for_opaque_pixels(renderer):
    _install_image(renderer, "A.bmp", _opaque_img(16, 16, (10, 20, 30)))
    renderer.blit("A", "A.bmp", (0, 0, 16, 16), (0, 0), 1.0)
    # All 16x16 canvas pixels should have provenance into slot A (atlas 0..15, 0..15).
    p = renderer.provenance
    assert (p[:16, :16, 0] >= 0).all()
    assert (p[:16, :16, 0] <= 15).all()
    assert (p[:16, :16, 1] <= 15).all()
    # Other canvas pixels are unset.
    assert (p[16:, :, 0] == -1).all()


def test_transparent_pixels_do_not_mark_provenance(renderer):
    _install_image(renderer, "A.bmp", _half_transparent(16, 16))
    renderer.blit("A", "A.bmp", (0, 0, 16, 16), (0, 0), 1.0)
    # Left half opaque -> provenance set. Right half transparent -> stays -1.
    p = renderer.provenance
    assert (p[:16, :8, 0] >= 0).all()
    assert (p[:16, 8:16, 0] == -1).all()


def test_overdraw_replaces_old_provenance(renderer):
    """Critical Codex finding: MAIN under controls must not be marked visible."""
    _install_image(renderer, "A.bmp", _opaque_img(16, 16, (255, 0, 0)))
    _install_image(renderer, "B.bmp", _opaque_img(8, 8, (0, 0, 255)))
    renderer.blit("A", "A.bmp", (0, 0, 16, 16), (0, 0), 1.0)
    # Overdraw center 8x8 with slot B.
    renderer.blit("B", "B.bmp", (0, 0, 8, 8), (4, 4), 1.0)
    p = renderer.provenance
    # Center pixels now reference slot B's atlas region (x in [16, 23]).
    assert (p[6, 6, 1] >= 16) and (p[6, 6, 1] <= 23)
    # Corner pixels still slot A.
    assert (p[0, 0, 1] >= 0) and (p[0, 0, 1] <= 15)
    # Visible atlas mask: A only covers the uncovered ring, B covers center.
    mask = np.array(renderer.visible_mask)
    a_visible = int(mask[:16, :16].sum() / 255)
    b_visible = int(mask[:16, 16:32].sum() / 255)
    # Slot A should NOT have all 256 atlas pixels visible (some covered by B).
    assert a_visible < 256, f"all of slot A still marked visible despite overdraw (got {a_visible}/256)"
    assert b_visible == 64, f"slot B should be fully visible (got {b_visible}/64)"


def test_fill_rect_clears_provenance(renderer):
    _install_image(renderer, "A.bmp", _opaque_img(16, 16, (10, 20, 30)))
    renderer.blit("A", "A.bmp", (0, 0, 16, 16), (0, 0), 1.0)
    assert (renderer.provenance[:16, :16, 0] >= 0).all()
    renderer.fill_rect((0, 0, 16, 16), (0, 0, 0, 255))
    # After fill_rect, the same canvas pixels should have provenance = -1.
    assert (renderer.provenance[:16, :16, 0] == -1).all()


def test_erase_clears_provenance(renderer):
    _install_image(renderer, "A.bmp", _opaque_img(16, 16, (10, 20, 30)))
    renderer.blit("A", "A.bmp", (0, 0, 16, 16), (0, 0), 1.0)
    renderer.erase_rect_with_edge_average((4, 4, 8, 8), scale=1.0)
    # The 8x8 erased rect should have cleared provenance.
    assert (renderer.provenance[4:12, 4:12, 0] == -1).all()
    # Surrounding pixels still slot A.
    assert (renderer.provenance[0, 0, 0] >= 0)


def test_scaled_blit_maps_to_correct_atlas_coords(renderer):
    """Scaled NEAREST blit: each canvas pixel's provenance must point at the
    correct source atlas pixel under PIL's NEAREST rule (i*sw // rendered_w)."""
    # Distinct per-pixel atlas content (RGB encodes (y, x)).
    arr = np.zeros((4, 4, 4), dtype=np.uint8)
    for y in range(4):
        for x in range(4):
            arr[y, x] = (y * 16, x * 16, 0, 255)
    _install_image(renderer, "A.bmp", Image.fromarray(arr, "RGBA"))
    renderer.blit("A", "A.bmp", (0, 0, 4, 4), (0, 0), 2.0)  # 2x scale
    # Rendered 8x8 region at (0,0). For canvas pixel (cy=0, cx=0): src=(0,0) -> atlas (0,0).
    # Canvas pixel (cy=7, cx=7): src=(3, 3) -> atlas (3, 3).
    p = renderer.provenance
    assert tuple(p[0, 0]) == (0, 0)
    assert tuple(p[7, 7]) == (3, 3)
    assert tuple(p[2, 5]) == (1, 2)   # cy=2 -> src_j=1, cx=5 -> src_i=2


def _gradient_image(w: int, h: int) -> Image.Image:
    """Per-pixel-unique RGBA so we can recover source-coords from canvas RGB."""
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            arr[y, x] = (y, x, 0, 255)
    return Image.fromarray(arr, "RGBA")


def test_non_integer_scale_matches_pil_nearest(renderer):
    """For a non-integer scale, the provenance dest atlas coords for every
    canvas pixel must match what PIL's NEAREST resize actually picked. This is
    Codex's specific concern: `i * sw // rendered_w` does NOT in general match
    PIL NEAREST for fractional scales; we now resize a packed atlas-id image
    through the same PIL path."""
    img = _gradient_image(16, 16)
    _install_image(renderer, "A.bmp", img)
    renderer.blit("A", "A.bmp", (0, 0, 16, 16), (0, 0), 1.5)  # non-integer
    rendered_w = max(1, round(16 * 1.5))
    rendered_h = max(1, round(16 * 1.5))
    # Compute what PIL NEAREST does on the source RGB image.
    expected_rendered = img.resize((rendered_w, rendered_h), Image.Resampling.NEAREST)
    rgb = np.array(expected_rendered)  # encodes (y, x, 0, 255)
    # For each rendered canvas pixel, the source y/x are stored in R/G channels.
    for cy in range(rendered_h):
        for cx in range(rendered_w):
            if cy >= renderer.canvas_h or cx >= renderer.canvas_w:
                continue
            src_y_expected = int(rgb[cy, cx, 0])
            src_x_expected = int(rgb[cy, cx, 1])
            atlas_y_expected = 0 + 0 + src_y_expected  # slot.y + sy + src_y
            atlas_x_expected = 0 + 0 + src_x_expected
            assert tuple(renderer.provenance[cy, cx]) == (atlas_y_expected, atlas_x_expected), (
                f"cy={cy} cx={cx}: provenance {tuple(renderer.provenance[cy, cx])} != "
                f"PIL expected ({atlas_y_expected}, {atlas_x_expected})"
            )


def test_component_sx_sy_provenance(renderer):
    """Asymmetric x/y scales: provenance must still match PIL's separate
    per-axis sampling, not be skewed."""
    img = _gradient_image(16, 16)
    _install_image(renderer, "A.bmp", img)
    renderer.blit("A", "A.bmp", (0, 0, 16, 16), (0, 0), (2.0, 1.0))  # 2x wider, same height
    expected_rendered = img.resize((32, 16), Image.Resampling.NEAREST)
    rgb = np.array(expected_rendered)
    for cy in range(16):
        for cx in range(32):
            if cy >= renderer.canvas_h or cx >= renderer.canvas_w:
                continue
            sy = int(rgb[cy, cx, 0]); sx = int(rgb[cy, cx, 1])
            assert tuple(renderer.provenance[cy, cx]) == (sy, sx), (
                f"asymmetric scale: cy={cy} cx={cx} prov={tuple(renderer.provenance[cy, cx])} expected=({sy},{sx})"
            )


def test_procedural_overlap_clears_atlas_provenance(renderer):
    """If a procedural (non-atlas) layer is composited over an existing atlas
    blit, the canvas pixels under the opaque procedural region must lose
    their atlas provenance."""
    _install_image(renderer, "A.bmp", _opaque_img(16, 16, (255, 0, 0)))
    renderer.blit("A", "A.bmp", (0, 0, 16, 16), (0, 0), 1.0)
    # Procedural layer: opaque rectangle in canvas-space.
    layer = Image.new("RGBA", (8, 8), (0, 255, 0, 255))
    renderer.composite_procedural(layer, (4, 4))
    # Center 8x8 should now be procedural -> provenance cleared.
    assert (renderer.provenance[4:12, 4:12, 0] == -1).all()
    # Outside the procedural rect, atlas provenance survives.
    assert renderer.provenance[0, 0, 0] >= 0


def test_window_pass_clears_target_when_subwindow_opaque_without_provenance():
    """Codex finding 2: when sub-canvas is opaque but has no atlas provenance
    (procedural text/histogram drawn in the sub-window), the main provenance
    at those locations must be CLEARED, not kept.

    Exercises the actual `_render_window_pass` rather than re-implementing
    the paste logic.
    """
    r = cli.Renderer(skin_source=ROOT / "assets" / "default_skin", canvas_w=32, canvas_h=32)
    r.slots = {
        "A": Slot(id=0, name="A", file="A.bmp", x=0, y=0, w=8, h=8, loss_weight=1.0),
    }
    _install_image(r, "A.bmp", _opaque_img(8, 8, (255, 0, 0)))

    # Pre-paint main canvas at the target window region with atlas A provenance.
    r.blit("A", "A.bmp", (0, 0, 8, 8), (4, 4), 1.0)
    assert (r.provenance[4:12, 4:12, 0] >= 0).all()

    # Inner "render function" that runs during the window pass: only paints a
    # procedural (opaque, non-atlas) rectangle on the active canvas. This
    # mimics e.g. a histogram or text overlay drawn inside one window.
    def procedural_inner(renderer, params):
        # The active canvas here is the sub-canvas the pass swapped in.
        renderer.fill_rect((6, 6, 4, 4), (0, 255, 0, 255))
        # Re-mark it as opaque via direct paint (fill_rect already does this
        # but we want to also make sure non-blit opaque pixels exit through
        # the paste logic).
        from PIL import ImageDraw as _ID
        _ID.Draw(renderer.canvas).rectangle([6, 6, 9, 9], fill=(0, 255, 0, 255))

    cli._render_window_pass(
        r, procedural_inner, {"windows": {"sub": [0, 0]}},
        window_name="sub", final_origin=(0, 0), window_sy=1.0,
        canvas_w=32, canvas_h=32,
    )

    # The procedural rectangle painted over (6..9, 6..9). Their main
    # provenance must now be -1 (procedural overlay, not atlas).
    assert (r.provenance[6:10, 6:10, 0] == -1).all(), (
        "opaque procedural sub-canvas pixels did not clear main provenance"
    )
    # Pixels outside the procedural rect (still within the original blit's
    # 4..12 region) keep slot A.
    assert r.provenance[4, 4, 0] >= 0
    assert r.provenance[11, 11, 0] >= 0


def test_visible_mask_reflects_only_surviving_pixels(renderer):
    """End-to-end: cover MAIN with B, fill part of B's region, verify final
    visible_atlas_mask is the SHRUNK set of surviving pixels in each slot."""
    _install_image(renderer, "A.bmp", _opaque_img(16, 16, (255, 0, 0)))
    _install_image(renderer, "B.bmp", _opaque_img(16, 16, (0, 0, 255)))
    renderer.blit("A", "A.bmp", (0, 0, 16, 16), (0, 0), 1.0)
    renderer.blit("B", "B.bmp", (0, 0, 8, 8), (0, 0), 1.0)
    renderer.fill_rect((0, 0, 4, 4), (0, 0, 0, 255))  # erase a chunk of B
    mask = np.array(renderer.visible_mask)
    a_visible = int(mask[:16, :16].sum() / 255)
    b_visible = int(mask[:16, 16:32].sum() / 255)
    # A was fully drawn (256), then 8x8 of it was covered by B -> 256 - 64 = 192.
    assert a_visible == 256 - 64, f"A visible={a_visible}, expected 192"
    # B's 8x8 had some pixels cleared by the fill. Exact count depends on PIL's
    # inclusive-rectangle behaviour, but it must be strictly less than 64.
    assert 0 < b_visible < 64, f"B visible={b_visible}, expected strictly between 0 and 64"
