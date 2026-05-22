#!/usr/bin/env python3
"""Render a V6 training dataset: views + per-sample provenance labels.

For one skin source, render K randomized variants and write:

    out/views/{skin_id}_{variant:04d}.png        (RGB view)
    out/labels/{skin_id}_{variant:04d}.npz       (per-file visible_mask, uv_target)
    out/train.csv                                (skin_id, variant_id, view_png, labels_npz)

The render seed is stable_seed(skin_id, variant_id) so the views are bit-exact
reproducible with scripts/02_render_dataset.py for the same (skin_id, variant_id)
pair.

Used to bootstrap Stage 2 (copy head only, one skin) of the V6 source-preserving
training plan.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atlas_ai.v6_labels import build_v6_labels, save_v6_labels


def stable_seed(skin_id: str, variant_id: int) -> int:
    """Identical seed function to scripts/02_render_dataset.py."""
    base = int(hashlib.sha256(skin_id.encode("utf-8")).hexdigest()[:12], 16)
    return base * 1_000_003 + variant_id


def _load_cli_module():
    spec = importlib.util.spec_from_file_location(
        "cranamp_cli_tool",
        ROOT / "cranamp_cli/cranamp/tools/cranamp_cli.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def render_one_with_labels(
    cm,
    skin_source: Path,
    skin_id: str,
    variant_id: int,
    canvas_w: int,
    canvas_h: int,
    views_dir: Path,
    labels_dir: Path,
) -> tuple[Path, Path]:
    sample_id = f"{skin_id}_{variant_id:04d}"
    seed = stable_seed(skin_id, variant_id)
    params = cm.rand_params(seed=seed, canvas_w=canvas_w, canvas_h=canvas_h)
    renderer = cm.render_with_params(skin_source, params, canvas_w, canvas_h)

    view_path = views_dir / f"{sample_id}.png"
    labels_path = labels_dir / f"{sample_id}.npz"
    views_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    renderer.canvas.convert("RGB").save(view_path)
    labels = build_v6_labels(renderer.provenance, canvas_w, canvas_h)
    save_v6_labels(labels, labels_path)
    return view_path, labels_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skin-source", required=True,
                        help="Directory or .wsz path containing the source BMPs")
    parser.add_argument("--skin-id", required=True,
                        help="Stable id used for seeding (and CSV column).")
    parser.add_argument("--variants", type=int, default=32)
    parser.add_argument("--canvas-w", type=int, default=960)
    parser.add_argument("--canvas-h", type=int, default=1728)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cm = _load_cli_module()
    out = Path(args.out)
    views_dir = out / "views"
    labels_dir = out / "labels"
    out.mkdir(parents=True, exist_ok=True)
    skin_source = Path(args.skin_source)

    csv_path = out / "train.csv"
    total_visible_by_file: dict[str, int] = {}
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["skin_id", "variant_id", "view_png", "labels_npz"])
        for variant_id in range(args.variants):
            view_path, labels_path = render_one_with_labels(
                cm,
                skin_source=skin_source,
                skin_id=args.skin_id,
                variant_id=variant_id,
                canvas_w=args.canvas_w,
                canvas_h=args.canvas_h,
                views_dir=views_dir,
                labels_dir=labels_dir,
            )
            labels = np.load(labels_path)
            for key in labels.files:
                if not key.startswith("visible_"):
                    continue
                total_visible_by_file[key.removeprefix("visible_")] = (
                    total_visible_by_file.get(key.removeprefix("visible_"), 0)
                    + int(labels[key].sum())
                )
            labels.close()
            writer.writerow([
                args.skin_id,
                f"{variant_id:04d}",
                str(view_path),
                str(labels_path),
            ])

    print(f"wrote {args.variants} sample(s) to {out}")
    print(f"  views:  {views_dir}")
    print(f"  labels: {labels_dir}")
    print(f"  csv:    {csv_path}")
    if total_visible_by_file:
        print("  visible pixels by slot:")
        for slot, count in sorted(total_visible_by_file.items()):
            print(f"    {slot}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
