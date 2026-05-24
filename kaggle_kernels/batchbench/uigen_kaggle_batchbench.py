"""V7 Gate B-skin batch-size sweep on Kaggle Tesla T4.

The 80 000-step Gate B-skin run reached only ~5.7k samples/skin (14 skins,
batch=1). Loss was still descending and skin conditioning helped every file.
Before changing architecture or training longer, we need to know how big the
same-file batch can go on T4 with the c48, file_embedding_dim=32,
skin_embedding_dim=64 model.

This kernel runs scripts/benchmark_runtime.py three times against the
gateB_skin experiment YAML, overriding --batch to 4 / 8 / 14 and using
--within-file-replacement false so each batch covers distinct skins (the
intended continuation recipe).

We capture sec/step + peak VRAM for each run, plus the run/exit code, so
we can pick the largest batch that fits comfortably (VRAM headroom and
no OOM) and gives the best throughput.

Outputs: /kaggle/working/batchbench_outputs/{NN_label}.stdout.txt,
plus summary.json with the parsed numbers.
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
OUT = WORK / "batchbench_outputs"
OUT.mkdir(parents=True, exist_ok=True)

BENCH_STEPS = 300            # plenty for a sec/step median; ~1-2 min per batch on T4
BATCHES_TO_TRY = [4, 8, 14]  # 14 = num_skins, so every batch covers every skin once


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
    """Pull the per-step time and peak VRAM out of benchmark_runtime stdout."""
    sec_per_step = None
    peak_vram_mib = None
    # benchmark_runtime prints lines like:
    #   sec/step median:  0.0451
    #   peak VRAM:        1234 MiB
    for line in stdout.splitlines():
        m = re.search(r"sec/step\s+median\s*:\s*([0-9.]+)", line)
        if m:
            sec_per_step = float(m.group(1))
        m = re.search(r"peak\s+VRAM\s*:\s*([0-9]+)\s*MiB", line)
        if m:
            peak_vram_mib = int(m.group(1))
    return {"sec_per_step_median": sec_per_step, "peak_vram_mib": peak_vram_mib}


summaries: list[dict] = []

# 1. Repo + deps.
summaries.append(run("01_clone", ["git", "clone", "--branch", REPO_BRANCH, REPO_URL, str(REPO)]))
summaries.append(run("02_rev", ["git", "-C", str(REPO), "rev-parse", "HEAD"]))
summaries.append(run("03_pip", [sys.executable, "-m", "pip", "install", "--quiet",
                                "safetensors", "pyyaml"]))

env_for_run = {
    "UIGEN_RUNTIME": "configs/runtime/kaggle.yaml",
    "PYTHON": sys.executable,
}
summaries.append(run("04_print_env", [sys.executable, "scripts/print_env.py"],
                     cwd=REPO, env=env_for_run))

# 2. Sweep. Each run is independent.
sweep_results: list[dict] = []
for batch in BATCHES_TO_TRY:
    label = f"05_bench_batch{batch:02d}"
    res = run(label, [
        sys.executable, "scripts/benchmark_runtime.py",
        "--experiment", "experiments/v7_completer_gateB_skin.yaml",
        "--steps", str(BENCH_STEPS),
        "--batch", str(batch),
        "--within-file-replacement", "false",
    ], cwd=REPO, env=env_for_run, timeout=15 * 60)
    summaries.append({k: v for k, v in res.items() if k != "stdout" and k != "stderr"})
    parsed = parse_bench(res["stdout"])
    sweep_results.append({
        "batch": batch,
        "rc": res["rc"],
        "duration_sec": res["dur_sec"],
        **parsed,
    })

# 3. Recommend a batch: largest one that didn't OOM (rc==0) and gives a sensible
# sec/step. Headroom rule: peak_vram <= 12 GiB on a 15-GiB T4 leaves cushion for
# allocator fragmentation; bigger is fine but report it.
def _ok(row: dict) -> bool:
    return row["rc"] == 0 and row.get("sec_per_step_median") is not None

valid = [r for r in sweep_results if _ok(r)]
if valid:
    # Best throughput in samples/sec = batch / sec_per_step
    for r in valid:
        r["samples_per_sec"] = round(r["batch"] / r["sec_per_step_median"], 2) \
            if r["sec_per_step_median"] else None
    valid.sort(key=lambda r: (-r["samples_per_sec"] or 0, -r["batch"]))
    chosen = valid[0]
else:
    chosen = None

(OUT / "summary.json").write_text(json.dumps({
    "steps": summaries,
    "sweep": sweep_results,
    "recommended": chosen,
}, indent=2))

print("\n=== batchbench sweep ===", flush=True)
print(json.dumps(sweep_results, indent=2), flush=True)
print("\n=== recommended ===", flush=True)
print(json.dumps(chosen, indent=2), flush=True)

all_ok = all(s["rc"] == 0 for s in summaries)
print("BATCHBENCH_RESULT:", "PASS" if all_ok else "FAIL", flush=True)
if not all_ok:
    raise SystemExit(1)
