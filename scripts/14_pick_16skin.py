#!/usr/bin/env python3
"""Pick a diverse 16-skin set for V4 Gate 3 memorization.

Strategy:
- Sample a wide candidate pool of `.wsz` skins from `skins_raw/`.
- Profile each skin's `MAIN.bmp` on five normalized features: brightness,
  contrast, palette size, saturation, and edge density.
- Anchor the selection on the three Gate 2 skins (darkside, Aguileramp,
  Zelda) and then greedily pick the remaining 13 by farthest-point
  sampling in the normalized feature space.

The script is deterministic for a given pool sample seed and pool size,
and emits JSON with the chosen skins plus their per-feature stats so the
candidate report can be reproduced or updated later.
"""
from __future__ import annotations

import argparse
import io
import json
import random
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]

ANCHORS = (
    "darkside.wsz",
    "Aguileramp_-_OldSchool.wsz",
    "Zelda_Amp_Gold.wsz",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skins-dir", default="skins_raw")
    parser.add_argument("--pool", type=int, default=500,
                        help="Random subset of skins to consider (excluding anchors).")
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--out-json", default="reports/v4_gate3_candidates/picks.json")
    parser.add_argument("--out-list", default="reports/v4_gate3_candidates/picks.txt")
    args = parser.parse_args()

    skins_dir = Path(args.skins_dir)
    if not skins_dir.is_absolute():
        skins_dir = REPO / skins_dir
    all_skins = sorted(skins_dir.glob("*.wsz"))
    if not all_skins:
        print(f"no skins under {skins_dir}", file=sys.stderr)
        return 1

    anchor_paths = []
    for name in ANCHORS:
        candidate = skins_dir / name
        if not candidate.exists():
            print(f"anchor missing: {candidate}", file=sys.stderr)
            return 2
        anchor_paths.append(candidate)

    rng = random.Random(args.seed)
    non_anchor = [p for p in all_skins if p.name not in ANCHORS]
    if len(non_anchor) > args.pool:
        non_anchor = rng.sample(non_anchor, args.pool)

    profiles = []
    for path in anchor_paths + non_anchor:
        try:
            profile = profile_skin(path)
        except Exception as exc:  # noqa: BLE001
            profile = {"skin": path.name, "error": str(exc)}
        profiles.append(profile)

    valid_all = [p for p in profiles if "error" not in p]

    # Minimum-viability filter: keep palette/contrast above degenerate floor.
    # Anchors bypass the filter so Gate 2 results stay comparable.
    def viable(p: dict) -> bool:
        if p["skin"] in ANCHORS:
            return True
        if p["palette_estimate"] < 30:
            return False
        if p["L_std"] < 20:
            return False
        return True

    valid = [p for p in valid_all if viable(p)]
    print(f"profiled {len(valid_all)} skins, kept {len(valid)} after viability filter",
          file=sys.stderr)
    if len(valid) < args.count:
        print(f"only {len(valid)} valid skins, need {args.count}", file=sys.stderr)
        return 3

    # Normalize features to [0, 1] before farthest-point sampling.
    feature_keys = ("L_mean", "L_std", "saturation", "log_palette", "edge_density")
    feat_matrix = np.array([[p[k] for k in feature_keys] for p in valid], dtype=np.float64)
    feat_min = feat_matrix.min(axis=0)
    feat_max = feat_matrix.max(axis=0)
    feat_range = np.where(feat_max - feat_min > 0, feat_max - feat_min, 1.0)
    feat_norm = (feat_matrix - feat_min) / feat_range
    name_to_idx = {p["skin"]: idx for idx, p in enumerate(valid)}

    selected = []
    for anchor in ANCHORS:
        idx = name_to_idx.get(anchor)
        if idx is None:
            print(f"anchor {anchor} did not profile cleanly", file=sys.stderr)
            return 4
        selected.append(idx)

    while len(selected) < args.count:
        sel_arr = feat_norm[selected]
        dists = np.full(len(valid), np.inf)
        for vector in sel_arr:
            d = np.linalg.norm(feat_norm - vector, axis=1)
            dists = np.minimum(dists, d)
        for i in selected:
            dists[i] = -1.0
        next_idx = int(np.argmax(dists))
        selected.append(next_idx)

    picks = [valid[idx] for idx in selected]
    for i, p in enumerate(picks):
        p["rank"] = i
        p["is_anchor"] = p["skin"] in ANCHORS
        if p["is_anchor"]:
            p["reason"] = "Gate 2 anchor"
        else:
            p["reason"] = describe_pick(p, feat_norm[selected[0]:selected[i]] if i > 0 else None,
                                        feat_norm[selected[i]])

    out_data = {
        "seed": args.seed,
        "pool_size": args.pool,
        "skins_considered": len(valid),
        "anchors": list(ANCHORS),
        "picks": picks,
        "feature_keys": list(feature_keys),
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out_data, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    out_list = Path(args.out_list)
    out_list.parent.mkdir(parents=True, exist_ok=True)
    out_list.write_text("\n".join(p["skin"] for p in picks) + "\n", encoding="utf-8")

    print(f"wrote {out_json}")
    print(f"wrote {out_list}")
    print()
    print(f"{'#':>2} {'skin':40s} {'L':>5s} {'std':>5s} {'sat':>5s} {'pal':>5s} {'edge':>5s}  reason")
    for p in picks:
        print(f"{p['rank']:>2} {p['skin'][:40]:40s} {p['L_mean']:5.1f} {p['L_std']:5.1f} "
              f"{p['saturation']:5.2f} {p['palette_estimate']:5d} {p['edge_density']:5.2f}  {p['reason']}")
    return 0


def profile_skin(path: Path) -> dict:
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        main = next((n for n in names if n.upper().endswith("MAIN.BMP")), None)
        if main is None:
            raise FileNotFoundError("MAIN.bmp not in archive")
        with zf.open(main) as fp:
            img = Image.open(io.BytesIO(fp.read())).convert("RGB")
    arr = np.asarray(img).astype(np.float32)
    if arr.shape[0] < 100 or arr.shape[1] < 200:
        raise ValueError(f"MAIN too small ({arr.shape})")
    flat = arr.astype(np.uint8).reshape(-1, 3)
    sample_step = max(1, len(flat) // 5000)
    uniq = len({tuple(row) for row in flat[::sample_step]})
    mx = arr.max(axis=2)
    mn = arr.min(axis=2)
    sat = float(((mx - mn) / (mx + 1e-6)).mean())
    gx = np.abs(np.diff(arr, axis=1)).mean()
    gy = np.abs(np.diff(arr, axis=0)).mean()
    edge = float((gx + gy) / 2.0 / 255.0)
    return {
        "skin": path.name,
        "main_size": [int(arr.shape[1]), int(arr.shape[0])],
        "L_mean": float(arr.mean()),
        "L_std": float(arr.std()),
        "saturation": sat,
        "palette_estimate": int(uniq),
        "log_palette": float(np.log10(max(uniq, 1))),
        "edge_density": edge,
    }


def describe_pick(p: dict, _prev_features, my_vector) -> str:
    """One-line label summarising what this skin contributes."""
    tags = []
    if p["L_mean"] < 60:
        tags.append("dark")
    elif p["L_mean"] > 180:
        tags.append("bright")
    else:
        tags.append("mid-tone")
    if p["L_std"] > 75:
        tags.append("high-contrast")
    elif p["L_std"] < 30:
        tags.append("low-contrast")
    if p["saturation"] > 0.5:
        tags.append("saturated")
    elif p["saturation"] < 0.1:
        tags.append("muted")
    if p["palette_estimate"] < 50:
        tags.append("low-palette")
    elif p["palette_estimate"] > 1500:
        tags.append("photographic-palette")
    if p["edge_density"] > 0.12:
        tags.append("busy-texture")
    elif p["edge_density"] < 0.04:
        tags.append("flat-ui")
    return "/".join(tags) if tags else "diversity pick"


if __name__ == "__main__":
    raise SystemExit(main())
