#!/usr/bin/env python3
"""Parallel generator for the V11 LEAN full-skin dataset.

Runs make_v10_bmp_expert_dataset.py per skin in a process pool (each into an
isolated temp dir to avoid CSV-append races), then merges into one dataset and
carves a DISJOINT held-skin split. Native-res (canvas-w 384). Emits progress to
stdout per the project rule. Idempotent-ish: skip skins already merged.

Usage:
  python scripts/gen_lean_dataset.py --canon data_v11_skins_all \
      --out data_v11_lean --held-out data_v11_lean_held --held-n 64 \
      --scale lean --canvas-w 384 --workers 16
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BMPS = ["BALANCE", "CBUTTONS", "EQMAIN", "MAIN", "MONOSTER", "PLAYPAUS",
        "PLEDIT", "POSBAR", "SHUFREP", "TITLEBAR", "VOLUME"]


def _gen_one(args: tuple[str, str, str, int]) -> tuple[str, str]:
    skin_dir, skin_id, scale, cw = args
    tmp = f"/tmp/genparts/{skin_id}"
    Path(tmp).mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [sys.executable, "scripts/make_v10_bmp_expert_dataset.py", "--skin", skin_dir,
         "--skin-id", skin_id, "--out", tmp, "--scale", scale, "--canvas-w", str(cw),
         "--progress-every", "0"],
        cwd=REPO, capture_output=True, text=True)
    return (skin_id, tmp if r.returncode == 0 else f"ERR:{r.stderr[-200:]}")


def _merge(part: Path, out: Path) -> None:
    for sub in ("renders", "states"):
        sp = part / sub
        if sp.is_dir():
            for f in sp.iterdir():
                shutil.move(str(f), str(out / sub / f.name))
    tp = part / "targets"
    if tp.is_dir():
        for skind in tp.iterdir():
            shutil.move(str(skind), str(out / "targets" / skind.name))
    for bmp in BMPS:
        src = part / "csv" / f"train_{bmp}.csv"
        if not src.exists():
            continue
        dst = out / "csv" / f"train_{bmp}.csv"
        rows = src.read_text().splitlines()
        if not dst.exists():
            dst.write_text("\n".join(rows) + "\n")
        else:
            with dst.open("a") as f:
                f.write("\n".join(rows[1:]) + "\n")  # skip header


def _prep_dirs(d: Path) -> None:
    for sub in ("renders", "states", "targets", "csv"):
        (d / sub).mkdir(parents=True, exist_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--canon", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--held-out", required=True)
    ap.add_argument("--held-n", type=int, default=64)
    ap.add_argument("--scale", default="lean")
    ap.add_argument("--canvas-w", type=int, default=384)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--progress-every", type=int, default=100)
    args = ap.parse_args()

    canon = Path(args.canon)
    skins = sorted(p.name for p in canon.iterdir() if p.is_dir() and not p.name.startswith("_"))
    held = skins[-args.held_n:]
    train = skins[:-args.held_n]
    print(f"gen_lean: {len(skins)} skins -> train {len(train)} / held {len(held)} "
          f"(scale={args.scale} cw={args.canvas_w} workers={args.workers})", flush=True)

    for name, group, dst in [("train", train, Path(args.out)), ("held", held, Path(args.held_out))]:
        _prep_dirs(dst)
        done = {r["skin_id"] for r in csv.DictReader((dst / "csv" / "train_CBUTTONS.csv").open())} \
            if (dst / "csv" / "train_CBUTTONS.csv").exists() else set()
        todo = [(str(canon / s), s, args.scale, args.canvas_w) for s in group if s not in done]
        print(f"[{name}] {len(group)} skins, {len(todo)} to gen (already {len(done)})", flush=True)
        t0 = time.time(); last = t0; n = 0; errs = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_gen_one, t) for t in todo]
            for fut in as_completed(futs):
                sid, res = fut.result()
                n += 1
                if res.startswith("ERR:"):
                    errs += 1
                    print(f"  WARN {sid}: {res}", flush=True)
                else:
                    _merge(Path(res), dst)
                    shutil.rmtree(res, ignore_errors=True)
                if args.progress_every and n % args.progress_every == 0:
                    now = time.time(); rate = args.progress_every / max(now - last, 1e-6)
                    eta = (len(todo) - n) / max(rate, 1e-6) / 60.0
                    print(f"  [{name} {n}/{len(todo)} {100.0*n/max(len(todo),1):4.1f}%] "
                          f"{rate:.1f} skin/s errs={errs} elapsed={(now-t0)/60.0:4.1f}min "
                          f"ETA={eta:4.1f}min", flush=True)
                    last = now
        rows = sum(1 for _ in (dst / "csv" / "train_CBUTTONS.csv").open()) - 1
        print(f"[{name}] DONE: {n} gen, {errs} errs, CBUTTONS rows={rows}, "
              f"{(time.time()-t0)/60.0:.1f}min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
