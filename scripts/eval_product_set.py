#!/usr/bin/env python3
"""Run the V8 mockup->skin baseline on a fixed product-eval image set.

Two renderers are scored per case:

  lite   = atlas_ai.torch_cranamp_renderer.render_visible(atlas)
           Differentiable Cranamp-LITE. It re-draws the extracted visible crops,
           so it matches the mockup almost by construction. OPTIMIZATION PROXY
           only -- it flatters the result and is NOT the product metric.

  cranamp = the real Cranamp CLI rendering the generated skin.wsz.
           This is the PRODUCT GATE: it exercises the actual engine's sprite
           semantics, so it reveals whether the atlas is truly a valid skin.
           cranamp_rgb_mae / cranamp_sobel_mae decide product progress.

Note (state confound, handled later): the real Cranamp render uses a fixed
--seed; its slider/toggle state will not match a mockup's drawn state, which
inflates the cranamp metric independently of asset quality. Neutralize that in a
later cycle; for now the cranamp render is still the honest product view.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from atlas_ai.hidden_state_compiler import compile_hidden_states
from atlas_ai.torch_cranamp_renderer import render_visible
from atlas_ai.v8_assets import image_to_tensor, save_exported_tensors, tensor_to_image
from atlas_ai.v8_layout import (
    default_layout,
    draw_layout_overlay,
    normalize_mockup_image,
    product_render_params,
    save_layout,
)
from atlas_ai.visible_extractor import extract_visible_assets

CRANAMP_CLI = REPO_ROOT / "cranamp_cli" / "cranamp-cli"
COMPONENTS = ("main", "eq", "playlist")


def _sobel(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        x = x.unsqueeze(0)
    kx = x.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
    ky = x.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)
    gx = F.conv2d(x, kx.repeat(x.shape[1], 1, 1, 1), padding=1, groups=x.shape[1])
    gy = F.conv2d(x, ky.repeat(x.shape[1], 1, 1, 1), padding=1, groups=x.shape[1])
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


def _metrics(rendered_t: torch.Tensor, target_t: torch.Tensor) -> tuple[float, float]:
    rgb = float((rendered_t - target_t).abs().mean())
    edge = float((_sobel(rendered_t) - _sobel(target_t)).abs().mean())
    return rgb, edge


def _side_by_side(a: Image.Image, b: Image.Image) -> Image.Image:
    out = Image.new("RGB", (a.width + b.width, max(a.height, b.height)), (8, 8, 10))
    out.paste(a.convert("RGB"), (0, 0))
    out.paste(b.convert("RGB"), (a.width, 0))
    return out


def _render_cranamp(skin_dir: Path, layout: dict, size: tuple[int, int],
                    params_path: Path, out_path: Path) -> Image.Image | None:
    """Render the generated skin through real Cranamp DETERMINISTICALLY.

    Uses `render-params` with neutral/product params derived from the layout
    (identity transforms, window_scales=1, neutral state, no jitter) so the
    engine draws the skin AS DESIGNED — the honest product gate. (render-random
    jitters geometry/state and is only fair seed-matched, so it is not used for
    product scoring.) Returns the rendered image, or None if it failed to load.
    """
    w, h = size
    params_path.write_text(json.dumps(product_render_params(layout), indent=2), encoding="utf-8")
    try:
        res = subprocess.run(
            [str(CRANAMP_CLI), "render-params", "--skin-dir", str(skin_dir),
             "--params-json", str(params_path), "--canvas-w", str(w), "--canvas-h", str(h),
             "--out-view", str(out_path)],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=180,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if res.returncode != 0 or not out_path.exists():
        return None
    img = Image.open(out_path).convert("RGB")
    if img.size != size:
        img = img.resize(size, Image.Resampling.LANCZOS)
        img.save(out_path)
    return img


def _crop(img: Image.Image, rect) -> Image.Image:
    x, y, w, h = (int(round(float(v))) for v in rect)
    return img.crop((x, y, x + w, y + h))


def _component_metrics(case: Path, normalized: Image.Image, cranamp_img: Image.Image,
                       layout: dict) -> dict:
    """Per-component (main/eq/playlist) cranamp rgb/sobel MAE + cropped SbS.

    The deterministic render places windows exactly at the layout rects, so the
    mockup and render crop to the same regions.
    """
    out = {}
    for name in COMPONENTS:
        rect = layout["rects"][name]
        m_crop = _crop(normalized, rect)
        r_crop = _crop(cranamp_img, rect)
        _side_by_side(m_crop, r_crop).save(case / f"component_{name}.png")
        rgb, edge = _metrics(image_to_tensor(r_crop), image_to_tensor(m_crop))
        out[f"{name}_cranamp_rgb_mae"] = rgb
        out[f"{name}_cranamp_sobel_mae"] = edge
    return out


def _iter_images(root: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in exts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mockups", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--default-skin", default="assets/default_skin")
    parser.add_argument("--seed", type=int, default=7,
                        help="Cranamp render-random seed (fixed state; see state-confound note).")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    mockups = _iter_images(Path(args.mockups))
    if args.limit is not None:
        mockups = mockups[: args.limit]
    if not mockups:
        raise SystemExit(f"no mockup images under {args.mockups}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in mockups:
        case = out / path.stem
        case.mkdir(parents=True, exist_ok=True)
        with Image.open(path) as source:
            normalized, _scale, _offset = normalize_mockup_image(source)
        layout = default_layout(*normalized.size)
        normalized.save(case / "normalized.png")
        draw_layout_overlay(normalized, layout).save(case / "debug_overlay.png")
        save_layout(layout, case / "layout.json")

        visible = extract_visible_assets(normalized, layout, default_skin=args.default_skin)
        compiled = compile_hidden_states(visible, default_skin=args.default_skin)
        skin_dir = case / "skin"
        zip_path = save_exported_tensors(compiled, skin_dir, default_skin=args.default_skin, package=True)
        target_t = image_to_tensor(normalized)

        # Proxy renderer (optimization only).
        lite_t = render_visible(compiled, layout)
        lite_img = tensor_to_image(lite_t)
        lite_img.save(case / "render_lite.png")
        _side_by_side(normalized, lite_img).save(case / "side_by_side_lite.png")
        lite_rgb, lite_edge = _metrics(lite_t, target_t)

        # Real Cranamp render (PRODUCT GATE), deterministic.
        cranamp_img = _render_cranamp(skin_dir, layout, normalized.size,
                                      case / "render_params.json", case / "render_cranamp.png")
        load_success = cranamp_img is not None
        comp = {}
        if load_success:
            cranamp_t = image_to_tensor(cranamp_img)
            cranamp_rgb, cranamp_edge = _metrics(cranamp_t, target_t)
            _side_by_side(normalized, cranamp_img).save(case / "side_by_side_cranamp.png")
            comp = _component_metrics(case, normalized, cranamp_img, layout)
        else:
            cranamp_rgb = cranamp_edge = float("nan")
            comp = {f"{n}_cranamp_{m}": float("nan")
                    for n in COMPONENTS for m in ("rgb_mae", "sobel_mae")}

        row = {
            "mockup": str(path),
            "case_dir": str(case),
            "skin_wsz": str(zip_path),
            "load_success": load_success,
            "cranamp_rgb_mae": cranamp_rgb,
            "cranamp_sobel_mae": cranamp_edge,
            **comp,
            "lite_rgb_mae": lite_rgb,
            "lite_sobel_mae": lite_edge,
            "human_similarity_1_5": "",
            "human_sharpness_1_5": "",
            "notes": "",
        }
        rows.append(row)
        if load_success:
            print(f"{path.name}: [PRODUCT cranamp] rgb={cranamp_rgb:.4f}  "
                  f"main={comp['main_cranamp_rgb_mae']:.4f} eq={comp['eq_cranamp_rgb_mae']:.4f} "
                  f"playlist={comp['playlist_cranamp_rgb_mae']:.4f}  (lite={lite_rgb:.4f})", flush=True)
        else:
            print(f"{path.name}: cranamp render FAILED (load_success=False)  "
                  f"(proxy lite rgb={lite_rgb:.4f})", flush=True)

    fields = list(rows[0].keys())
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (out / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    ok = [r for r in rows if r["load_success"]]
    print(f"\n{len(ok)}/{len(rows)} skins rendered in real Cranamp (deterministic).", flush=True)
    if ok:
        mean = lambda k: sum(r[k] for r in ok) / len(ok)
        print(f"PRODUCT mean cranamp_rgb_mae={mean('cranamp_rgb_mae'):.4f}  "
              f"(proxy lite_rgb_mae={mean('lite_rgb_mae'):.4f})", flush=True)
        print("  per-component cranamp_rgb_mae: "
              + "  ".join(f"{n}={mean(f'{n}_cranamp_rgb_mae'):.4f}" for n in COMPONENTS), flush=True)
    print(f"wrote {out / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
