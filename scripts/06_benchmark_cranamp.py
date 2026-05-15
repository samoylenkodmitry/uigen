#!/usr/bin/env python3
from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path
import subprocess


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cranamp-cli", default="cranamp_cli/cranamp-cli")
    parser.add_argument("--skin-dir", default="assets/default_skin")
    parser.add_argument("--renders", type=int, nargs="+", default=[100])
    parser.add_argument("--canvas-w", type=int, default=941)
    parser.add_argument("--canvas-h", type=int, default=1672)
    args = parser.parse_args()

    for render_count in args.renders:
        start = time.perf_counter()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for idx in range(render_count):
                cmd = [
                    args.cranamp_cli,
                    "render-random",
                    "--skin-dir",
                    args.skin_dir,
                    "--seed",
                    str(idx),
                    "--canvas-w",
                    str(args.canvas_w),
                    "--canvas-h",
                    str(args.canvas_h),
                    "--out-view",
                    str(root / f"{idx}.png"),
                    "--out-rects",
                    str(root / f"{idx}.rects.f32"),
                    "--out-state",
                    str(root / f"{idx}.state.f32"),
                    "--out-visible-atlas-mask",
                    str(root / f"{idx}.mask.png"),
                    "--out-params",
                    str(root / f"{idx}.params.json"),
                    "--state-balanced",
                    "false",
                ]
                subprocess.run(cmd, check=True)
        elapsed = time.perf_counter() - start
        print(f"{render_count} renders: {elapsed:.3f}s ({render_count / elapsed:.2f} renders/s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
