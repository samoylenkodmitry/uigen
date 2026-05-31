#!/usr/bin/env python3
"""Pre-decode dataset render PNGs to uint8 .npy (one-time), so training reads
arrays instead of decoding PNG every __getitem__.

PNG decode of native-res renders is the data-loading bottleneck on big GPUs
(measured: L40S only 3.4x a T4 because it's data-starved, not compute-bound).
The .npy cache is built ONCE and reused across every epoch and all 11 component
trainings (renders are shared). Enable with train_bmp_expert.py --fast-renders.

Idempotent + resumable: skips renders already packed. Emits progress per the
project's stdout rule.

Usage: python scripts/prepack_renders.py --data data_full --workers 16
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image


def _pack_one(args: tuple[Path, Path]) -> int:
    src, dst = args
    if dst.exists():
        return 0
    try:
        with Image.open(src) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)  # [H,W,3]
        tmp = dst.with_name(dst.stem + "__tmp.npy")   # ends in .npy so np.save won't re-append
        np.save(tmp, arr)
        tmp.replace(dst)
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"  WARN failed {src.name}: {e}", flush=True)
        return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="Dataset dir (has renders/).")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--progress-every", type=int, default=2000)
    args = ap.parse_args()

    root = Path(args.data)
    rdir = root / "renders"
    odir = root / "renders_npy"
    odir.mkdir(exist_ok=True)
    pngs = sorted(rdir.glob("*.png"))
    jobs = [(p, odir / (p.stem + ".npy")) for p in pngs]
    todo = [j for j in jobs if not j[1].exists()]
    print(f"prepack_renders: {len(pngs)} renders, {len(todo)} to pack -> {odir} "
          f"(workers={args.workers})", flush=True)
    t0 = time.time(); last = t0; done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, n in enumerate(ex.map(_pack_one, todo), 1):
            done += n
            if args.progress_every and i % args.progress_every == 0:
                now = time.time()
                rate = args.progress_every / max(now - last, 1e-6)
                eta = (len(todo) - i) / max(rate, 1e-6) / 60.0
                print(f"[{i}/{len(todo)}  {100.0*i/len(todo):5.1f}%]  {rate:.0f} render/s  "
                      f"elapsed={(now-t0)/60.0:5.1f}min  ETA={eta:5.1f}min", flush=True)
                last = now
    print(f"prepack done: packed {done}, total {len(pngs)} in {(time.time()-t0)/60.0:.1f}min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
