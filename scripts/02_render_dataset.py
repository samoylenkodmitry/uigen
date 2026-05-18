#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import subprocess


def stable_seed(skin_id: str, variant_id: int) -> int:
    base = int(hashlib.sha256(skin_id.encode("utf-8")).hexdigest()[:12], 16)
    return base * 1_000_003 + variant_id


def _render_one(args_tuple: tuple[str, str, int, str, int, int, str]) -> tuple[str, int]:
    cranamp_cli, skin_id, variant_id, source_path, canvas_w, canvas_h, out_str = args_tuple
    out = Path(out_str)
    sample_id = f"{skin_id}_{variant_id:04d}"
    seed = stable_seed(skin_id, variant_id)
    cmd = [
        cranamp_cli,
        "render-random",
        "--skin-dir", source_path,
        "--seed", str(seed),
        "--canvas-w", str(canvas_w),
        "--canvas-h", str(canvas_h),
        "--out-view", str(out / "views" / f"{sample_id}.png"),
    ]
    result = subprocess.run(cmd, capture_output=True)
    return (sample_id, result.returncode)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-skins", default="data_v35/valid_skins.csv")
    parser.add_argument("--cranamp-cli", default="cranamp_cli/cranamp-cli")
    parser.add_argument("--out", default="data_v35")
    parser.add_argument("--variants", type=int, default=4)
    parser.add_argument("--canvas-w", type=int, default=960)
    parser.add_argument("--canvas-h", type=int, default=1728)
    parser.add_argument("--limit", type=int, default=None,
                        help="Only render the first N skins from valid_skins.csv (for smoke).")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1),
                        help="Parallel cranamp-cli subprocesses. Defaults to ncpu-1.")
    args = parser.parse_args()

    out = Path(args.out)
    (out / "views").mkdir(parents=True, exist_ok=True)

    with Path(args.valid_skins).open("r", newline="", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row.get("status") == "ok"]
    if args.limit is not None:
        rows = rows[: args.limit]

    jobs = [
        (args.cranamp_cli, row["skin_id"], variant_id, row["source_path"],
         args.canvas_w, args.canvas_h, str(out))
        for row in rows for variant_id in range(args.variants)
    ]
    total = len(jobs)
    print(f"rendering {total} sample(s) across {args.workers} worker(s)...")

    failed: list[str] = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for sample_id, rc in pool.map(_render_one, jobs, chunksize=4):
            done += 1
            if rc != 0:
                failed.append(sample_id)
            if done % 200 == 0 or done == total:
                print(f"  {done}/{total} ({len(failed)} failed)")

    print(f"done. {total - len(failed)} ok, {len(failed)} failed")
    if failed:
        print(f"first failures: {failed[:5]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
