#!/usr/bin/env python3
"""Drive the V11 LEAN full-skin train of all 11 components within a HARD global
GPU-minute budget (so it never overshoots the free-credit allotment).

Each component trains L1 + Sobel/Laplacian + paired color-aug, --fast-renders,
--amp, resumable (--resume + checkpoint-every), with a per-component --max-minutes
slice = remaining_budget / remaining_components. Batch is chosen by atlas area so
big atlases (EQMAIN) don't OOM mid-run (wasted credits). Per-component cond_eval on
the held set gives a quality read. Stops nothing itself — caller stops the studio.

Usage:
  python scripts/train_all_lean.py --data data_v11_lean --held data_v11_lean_held \
      --out runs/v11_lean --budget-min 540 --device cuda
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from atlas_ai.export_spec import TRAINABLE_EXPORT_SPECS  # noqa: E402

# easy -> hard (Gate-1 experience); small atlases first so the cheap wins land early.
ORDER = ["POSBAR", "PLAYPAUS", "MONOSTER", "SHUFREP", "VOLUME", "BALANCE",
         "CBUTTONS", "TITLEBAR", "MAIN", "PLEDIT", "EQMAIN"]
SPEC = {s.file_name: s for s in TRAINABLE_EXPORT_SPECS}


def _batch_for(area: int) -> int:
    if area <= 8000:      # CBUTTONS(4896), small strips
        return 32
    if area <= 35000:     # MAIN(31900), TITLEBAR, VOLUME...
        return 16
    return 8              # EQMAIN(86625), PLEDIT


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--held", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget-min", type=float, default=540.0, help="HARD total GPU-min cap.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--min-slice", type=float, default=8.0, help="Skip a component if < this many min left.")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    comps = [c for c in ORDER if f"{c}.bmp" in SPEC]
    print(f"train_all_lean: {len(comps)} comps, budget {args.budget_min:.0f} GPU-min, "
          f"data={args.data}", flush=True)

    for i, stem in enumerate(comps):
        elapsed = (time.time() - t0) / 60.0
        remaining = args.budget_min - elapsed
        left = len(comps) - i
        if remaining < args.min_slice:
            print(f"BUDGET SPENT at {stem} (elapsed {elapsed:.0f}min) — stopping.", flush=True)
            break
        slice_min = remaining / left
        spec = SPEC[f"{stem}.bmp"]
        batch = _batch_for(spec.h * spec.w)
        cdir = out / stem
        print(f"\n=== [{i+1}/{len(comps)}] {stem} ({spec.w}x{spec.h}) batch={batch} "
              f"slice={slice_min:.0f}min (elapsed {elapsed:.0f}/{args.budget_min:.0f}) ===", flush=True)
        cmd = [sys.executable, "train_bmp_expert.py", "--data", args.data, "--bmp", f"{stem}.bmp",
               "--out", str(cdir), "--steps", "100000000", "--batch", str(batch),
               "--base", "48", "--attn-dim", "256", "--dec-ch", "128", "--heads", "4",
               "--attn-layers", "2", "--query-div", "4", "--decoder", "progressive",
               "--color-aug", "--amp", "--fast-renders", "--lr", "3e-4",
               "--max-minutes", f"{slice_min:.1f}", "--resume", "--checkpoint-every", "1500",
               "--eval-every", "4000", "--eval-max-items", "192", "--progress-every", "300",
               "--num-workers", "14", "--device", args.device]
        subprocess.run(cmd, cwd=REPO, env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                                           **__import__("os").environ})
        # quality read on held (best ckpt)
        ck = cdir / "best.safetensors"
        if not ck.exists():
            ck = cdir / "last.safetensors"
        if ck.exists():
            r = subprocess.run([sys.executable, "scripts/cond_eval.py", "--data", args.held,
                                "--bmp", f"{stem}.bmp", "--checkpoint", str(ck), "--device", args.device],
                               cwd=REPO, capture_output=True, text=True)
            print((r.stdout or r.stderr).strip().splitlines()[-1] if (r.stdout or r.stderr).strip() else "(cond_eval no output)", flush=True)

    print(f"\ntrain_all_lean DONE: elapsed {(time.time()-t0)/60.0:.0f}min, ckpts in {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
