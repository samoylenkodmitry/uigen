"""V7 Gate B-skin c64 batch-size sweep on Kaggle T4.

c48 state_family-focused stalled at the user's pre-declared capacity-
justified threshold (state_family mae 0.042 / hit5 0.78). Before
committing to a multi-hour c64 from-scratch run, find the largest
stable batch on T4 under the c64 + state_family-focused recipe.

Sweep batch 4, 8, 12 (skipping 14 since c64 ~2x VRAM vs c48 and
batchbench-c48 used 7.3 GiB at batch 14). 300 steps per batch is
plenty for a sec/step median; ~2 min each on T4. Total ~6-10 min.

Reads experiments/v7_completer_gateB_skin_c64.yaml so it uses the
exact training recipe: c64, state_family-focused mask mix, doubled
strip weights, skin_embedding_dim=64, sobel 0.25, within-file-
replacement=false.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_URL = "https://github.com/samoylenkodmitry/uigen.git"
REPO_BRANCH = "main"
WORK = Path("/kaggle/working")
REPO = WORK / "uigen"
OUT = WORK / "c64_batchbench_outputs"
OUT.mkdir(parents=True, exist_ok=True)

BENCH_STEPS = 300
BATCHES_TO_TRY = [4, 8, 12]


def run(label: str, cmd: list[str], cwd: Path | None = None, env: dict | None = None,
        timeout: int | None = None) -> dict:
    print(f"\n=== {label} ===", flush=True)
    print("$ " + " ".join(cmd), flush=True)
    t0 = time.time()
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    res = subprocess.run(cmd, cwd=cwd, env=full_env, capture_output=True,
                         text=True, timeout=timeout)
    dur = time.time() - t0
    (OUT / f"{label}.stdout.txt").write_text(res.stdout)
    (OUT / f"{label}.stderr.txt").write_text(res.stderr)
    print(res.stdout[-3000:], flush=True)
    if res.returncode != 0:
        print(res.stderr[-3000:], flush=True)
    return {"label": label, "rc": res.returncode, "dur_sec": round(dur, 1),
            "stdout": res.stdout, "stderr": res.stderr}


def parse_bench(stdout: str) -> dict:
    sec_per_step = None
    peak_vram_mib = None
    for line in stdout.splitlines():
        m = re.search(r"sec/step\s+median\s*:\s*([0-9.]+)", line)
        if m:
            sec_per_step = float(m.group(1))
        m = re.search(r"peak\s+VRAM\s*:\s*([0-9]+)\s*MiB", line)
        if m:
            peak_vram_mib = int(m.group(1))
    return {"sec_per_step_median": sec_per_step, "peak_vram_mib": peak_vram_mib}


summaries: list[dict] = []
summaries.append(run("01_clone", ["git", "clone", "--branch", REPO_BRANCH, REPO_URL, str(REPO)]))
summaries.append(run("02_rev", ["git", "-C", str(REPO), "rev-parse", "HEAD"]))
summaries.append(run("03_pip", [sys.executable, "-m", "pip", "install", "--quiet",
                                "safetensors", "pyyaml"]))

env_for_run = {"UIGEN_RUNTIME": "configs/runtime/kaggle.yaml", "PYTHON": sys.executable}
summaries.append(run("04_print_env", [sys.executable, "scripts/print_env.py"],
                     cwd=REPO, env=env_for_run))

sweep_results: list[dict] = []
for batch in BATCHES_TO_TRY:
    label = f"05_bench_batch{batch:02d}"
    res = run(label, [
        sys.executable, "scripts/benchmark_runtime.py",
        "--experiment", "experiments/v7_completer_gateB_skin_c64.yaml",
        "--steps", str(BENCH_STEPS),
        "--batch", str(batch),
        "--within-file-replacement", "false",
    ], cwd=REPO, env=env_for_run, timeout=15 * 60)
    summaries.append({k: v for k, v in res.items() if k not in ("stdout", "stderr")})
    parsed = parse_bench(res["stdout"])
    sweep_results.append({
        "batch": batch,
        "rc": res["rc"],
        "duration_sec": res["dur_sec"],
        **parsed,
    })

def _ok(row: dict) -> bool:
    return row["rc"] == 0 and row.get("sec_per_step_median") is not None

valid = [r for r in sweep_results if _ok(r)]
for r in valid:
    if r["sec_per_step_median"]:
        r["samples_per_sec"] = round(r["batch"] / r["sec_per_step_median"], 2)
    else:
        r["samples_per_sec"] = None

# Recommend the largest batch that fits comfortably (peak_vram <= 12 GiB)
# and has the best samples/sec. Ties broken by larger batch.
T4_HEADROOM_MIB = 12 * 1024
safe = [r for r in valid if (r.get("peak_vram_mib") or 0) <= T4_HEADROOM_MIB]
if safe:
    safe.sort(key=lambda r: (-(r["samples_per_sec"] or 0), -r["batch"]))
    chosen = safe[0]
else:
    chosen = None

(OUT / "summary.json").write_text(json.dumps({
    "steps": summaries,
    "sweep": sweep_results,
    "recommended": chosen,
    "headroom_mib": T4_HEADROOM_MIB,
}, indent=2))

print("\n=== c64 batchbench sweep ===", flush=True)
print(json.dumps(sweep_results, indent=2), flush=True)
print("\n=== recommended (≤12 GiB peak VRAM) ===", flush=True)
print(json.dumps(chosen, indent=2), flush=True)

all_ok = all(s["rc"] == 0 for s in summaries)
print("C64_BATCHBENCH_RESULT:", "PASS" if all_ok else "FAIL", flush=True)
if not all_ok:
    raise SystemExit(1)
