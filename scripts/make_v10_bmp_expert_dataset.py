#!/usr/bin/env python3
"""V10 BMP-expert dataset generator (deterministic state sweeps + random geometry).

For one source skin, render many full Cranamp views via render_with_params using
rand_params(seed, 960, 1728) and explicit `state` overrides per sweep family.
Component transforms / window scales stay randomized so each render also has
geometry distortion. The target for every variant is the skin's original BMP
(unchanged across variants). Per-BMP CSVs index the renders.

Writes (under --out):

    renders/<skin_id>_<vid>.png      [3, 1728, 960] real-Cranamp render
    states/<skin_id>_<vid>.json      full params + variant family label
    targets/<skin_id>/<FILE>.bmp     unchanged source BMP (11 trainable files)
    csv/train_<FILE>.csv             rows: render_png,target_bmp,skin_id,variant_id,state_json

Renderer-state gaps (state keys not exposed by the CLI) are logged into
`renderer_gaps.json` rather than silently skipped (handoff requirement).

Usage:

    python scripts/make_v10_bmp_expert_dataset.py \
        --skin assets/default_skin --skin-id default --scale smoke --out data_v10
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import shutil
import sys
import time as _time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "cranamp_cli" / "cranamp" / "tools"))

from atlas_ai.export_spec import TRAINABLE_EXPORT_SPECS
from cranamp_cli import rand_params, render_with_params  # noqa: E402


CANVAS_W, CANVAS_H = 960, 1728
TRAINABLE_NAMES = [s.file_name for s in TRAINABLE_EXPORT_SPECS]

# Per-scale variant caps per family. Schedule mirrors HANDOFF_V10's coverage plan.
SCALE_CAPS = {
    "smoke": dict(base=4, volume=4, balance=4, cbuttons=4, shufrep=4,
                  playpaus=3, posbar=4, eq_band=8, eq_random=4,
                  eq_onoff=2, pledit=4, extra=4),
    "gate1": dict(base=32, volume=28, balance=28, cbuttons=7, shufrep=16,
                  playpaus=3, posbar=29, eq_band=308, eq_random=96,
                  eq_onoff=4, pledit=64, extra=128),
    "gate2": dict(base=24, volume=28, balance=28, cbuttons=7, shufrep=16,
                  playpaus=3, posbar=29, eq_band=176, eq_random=48,
                  eq_onoff=4, pledit=32, extra=64),
}

# state keys the cranamp CLI exposes (verified via rand_params).
# pl_toggle and mono/stereo are NOT exposed -> recorded as renderer_gap.


def _override(params: dict, family: str, **state_kw) -> dict:
    """Clone params, set family label, overwrite given state keys."""
    p = copy.deepcopy(params)
    p["state"].update(state_kw)
    p["variant_family"] = family
    return p


def _plan(caps: dict) -> list[tuple[str, dict]]:
    """Yield (family, state-overrides) variants up to per-family caps."""
    plan: list[tuple[str, dict]] = []

    # base: random geometry, random state (no override)
    plan += [("base", {})] * caps["base"]

    # volume sweep (state["volume"] = i/27)
    for i in range(28):
        if len([1 for f, _ in plan if f == "volume"]) >= caps["volume"]:
            break
        plan.append(("volume", {"volume": i / 27.0}))

    # balance sweep
    for i in range(28):
        if len([1 for f, _ in plan if f == "balance"]) >= caps["balance"]:
            break
        plan.append(("balance", {"balance": i / 27.0}))

    # cbuttons: pressed_transport_button in {-1, 0..5}
    for v in (-1, 0, 1, 2, 3, 4, 5):
        if len([1 for f, _ in plan if f == "cbuttons"]) >= caps["cbuttons"]:
            break
        plan.append(("cbuttons", {"pressed_transport_button": v}))

    # shufrep: shuffle × repeat × eq_on × eq_auto
    sr = 0
    for sh in (False, True):
        for rp in (False, True):
            for eq_on in (False, True):
                for eq_auto in (False, True):
                    if sr >= caps["shufrep"]:
                        break
                    plan.append(("shufrep", {"shuffle": sh, "repeat": rp,
                                              "eq_on": eq_on, "eq_auto": eq_auto}))
                    sr += 1

    # playpaus: playback in {playing, paused, stopped}
    for v in ("playing", "paused", "stopped"):
        if len([1 for f, _ in plan if f == "playpaus"]) >= caps["playpaus"]:
            break
        plan.append(("playpaus", {"playback": v}))

    # posbar sweep
    for i in range(29):
        if len([1 for f, _ in plan if f == "posbar"]) >= caps["posbar"]:
            break
        plan.append(("posbar", {"posbar": i / 28.0}))

    # EQ single-band sweep (band j × position i)
    eb = 0
    for j in range(11):
        for i in range(28):
            if eb >= caps["eq_band"]:
                break
            eq_vals = [0.5] * 11
            eq_vals[j] = i / 27.0
            plan.append(("eq_band", {"eq_values": eq_vals}))
            eb += 1
        if eb >= caps["eq_band"]:
            break

    # EQ random curves: no override (rand_params already gives random eq_values).
    plan += [("eq_random", {})] * caps["eq_random"]

    # EQ on/off × auto: (eq_on, eq_auto) combos
    eo = 0
    for eq_on in (False, True):
        for eq_auto in (False, True):
            if eo >= caps["eq_onoff"]:
                break
            plan.append(("eq_onoff", {"eq_on": eq_on, "eq_auto": eq_auto}))
            eo += 1

    # PLEDIT scroll/selected sweeps (a coarse grid)
    pl = 0
    scrolls = [i / 7.0 for i in range(8)]
    rows = list(range(0, 18, 2))
    for s in scrolls:
        for r in rows:
            if pl >= caps["pledit"]:
                break
            plan.append(("pledit", {"playlist_scroll": s, "playlist_selected_row": r}))
            pl += 1
        if pl >= caps["pledit"]:
            break

    # extra random global mixtures
    plan += [("extra", {})] * caps["extra"]
    return plan


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skin", required=True, help="Source skin directory (with BMPs).")
    ap.add_argument("--skin-id", required=True, help="Stable id used in filenames.")
    ap.add_argument("--out", default="data_v10")
    ap.add_argument("--scale", default="smoke", choices=list(SCALE_CAPS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--append", action="store_true",
                    help="Append CSV rows instead of overwriting (multi-skin).")
    ap.add_argument("--progress-every", type=int, default=50)
    ap.add_argument("--canvas-w", type=int, default=CANVAS_W,
                    help="Render canvas width. Default 960 (skin upscaled ~3.3x). "
                         "Smaller (e.g. 384) renders the WHOLE UI at near-native "
                         "resolution = same info, ~10x cheaper to train.")
    ap.add_argument("--canvas-h", type=int, default=CANVAS_H)
    args = ap.parse_args()
    cw, ch = args.canvas_w, args.canvas_h

    skin_src = Path(args.skin)
    out = Path(args.out)
    (out / "renders").mkdir(parents=True, exist_ok=True)
    (out / "states").mkdir(parents=True, exist_ok=True)
    (out / "targets" / args.skin_id).mkdir(parents=True, exist_ok=True)
    (out / "csv").mkdir(parents=True, exist_ok=True)

    # Copy target BMPs (unchanged source) once per skin.
    for name in TRAINABLE_NAMES:
        src = skin_src / name
        if not src.exists():
            raise SystemExit(f"missing source BMP {src}")
        shutil.copy2(src, out / "targets" / args.skin_id / name)

    # Renderer-state gap log (handoff-required transparency).
    gaps_path = out / "renderer_gaps.json"
    gaps = {"unexposed_state_keys": ["mono_stereo", "pl_toggle"],
            "note": "rand_params/state does not control these; MONOSTER / pl-toggle sweeps skipped."}
    gaps_path.write_text(json.dumps(gaps, indent=2))

    plan = _plan(SCALE_CAPS[args.scale])
    print(f"V10 dataset: skin={args.skin_id} scale={args.scale} variants={len(plan)} "
          f"out={out} canvas={cw}x{ch}", flush=True)

    # Per-BMP CSVs (open in append mode if --append, else write headers).
    csv_files = {}
    csv_writers = {}
    for name in TRAINABLE_NAMES:
        p = out / "csv" / f"train_{Path(name).stem}.csv"
        is_new = not p.exists() or not args.append
        f = p.open("a" if args.append else "w", newline="", encoding="utf-8")
        w = csv.writer(f)
        if is_new and not args.append:
            w.writerow(["render_png", "target_bmp", "skin_id", "variant_id", "state_json"])
        elif is_new and args.append and p.stat().st_size == 0:
            w.writerow(["render_png", "target_bmp", "skin_id", "variant_id", "state_json"])
        csv_files[name] = f
        csv_writers[name] = w

    t0 = _time.monotonic()
    last_t = t0
    for vid, (family, overrides) in enumerate(plan):
        seed = args.seed * 10_000_000 + vid
        params = rand_params(seed, cw, ch)
        params = _override(params, family, **overrides)
        renderer = render_with_params(skin_src, params, cw, ch)
        vid_str = f"{vid:06d}"
        render_path = out / "renders" / f"{args.skin_id}_{vid_str}.png"
        state_path = out / "states" / f"{args.skin_id}_{vid_str}.json"
        renderer.canvas.convert("RGB").save(render_path)
        # params is JSON-safe (only native types from rand_params + our overrides).
        state_path.write_text(json.dumps(params, indent=2, sort_keys=True))
        for name in TRAINABLE_NAMES:
            tgt_rel = f"targets/{args.skin_id}/{name}"
            csv_writers[name].writerow([
                f"renders/{args.skin_id}_{vid_str}.png", tgt_rel,
                args.skin_id, vid_str, f"states/{args.skin_id}_{vid_str}.json",
            ])
        if args.progress_every > 0 and (vid + 1) % args.progress_every == 0:
            now = _time.monotonic()
            sps = (now - last_t) / args.progress_every
            eta = sps * max(len(plan) - (vid + 1), 0) / 60.0
            print(f"[{vid + 1:>5d}/{len(plan)}  {100.0 * (vid + 1) / len(plan):5.1f}%]  "
                  f"family={family:<10s} sec/render={sps:.2f}  elapsed={(now - t0) / 60.0:5.1f}min  "
                  f"ETA={eta:5.1f}min", flush=True)
            last_t = now

    for f in csv_files.values():
        f.close()
    print(f"done: {len(plan)} renders, 11 per-BMP CSVs at {out / 'csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
