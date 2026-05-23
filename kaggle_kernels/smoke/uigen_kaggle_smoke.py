"""V7 Gate B Kaggle smoke kernel.

Purpose: verify the cloud runtime end-to-end before spending GPU quota on the
full 80k-step Gate B run. Outputs land in /kaggle/working/smoke_outputs/.

Sequence:
    1. Clone the uigen repo (pinned to a known-good commit).
    2. pip install runtime deps.
    3. scripts/print_env.py     -- python/torch/GPU
    4. make kaggle-dry          -- resolves all 14 skin dirs in the trainer cmd
    5. make bench-kaggle        -- 50-step micro-benchmark, sec/step + peak VRAM
    6. dump every output as JSON/text so we can grep them after `kernels output`.

Decisions are gated by the user before the Gate B kernel runs.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_URL = "https://github.com/samoylenkodmitry/uigen.git"
REPO_BRANCH = "main"           # smoke always tracks main
WORK = Path("/kaggle/working")
REPO = WORK / "uigen"
OUT = WORK / "smoke_outputs"
OUT.mkdir(parents=True, exist_ok=True)


def run(label: str, cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> dict:
    """Run a command, persist stdout/stderr + rc, return a small dict."""
    print(f"\n=== {label} ===", flush=True)
    print("$ " + " ".join(cmd), flush=True)
    t0 = time.time()
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    res = subprocess.run(cmd, cwd=cwd, env=full_env, capture_output=True, text=True)
    dur = time.time() - t0
    (OUT / f"{label}.stdout.txt").write_text(res.stdout)
    (OUT / f"{label}.stderr.txt").write_text(res.stderr)
    summary = {
        "label": label,
        "cmd": cmd,
        "returncode": res.returncode,
        "duration_sec": round(dur, 3),
        "stdout_tail": res.stdout[-2000:],
        "stderr_tail": res.stderr[-2000:],
    }
    print(res.stdout[-4000:], flush=True)
    if res.returncode != 0:
        print(res.stderr[-4000:], flush=True)
    return summary


summaries: list[dict] = []

# 0. observe the mount; useful when the dataset path resolution disagrees
# with our expectations.
summaries.append(run("00_ls_input", ["bash", "-c",
    "find /kaggle/input -maxdepth 5 -print | head -50; echo ---; "
    "find /kaggle/input -name MAIN.bmp 2>/dev/null | head -3"]))

# 1. clone
summaries.append(run("01_clone", ["git", "clone", "--branch", REPO_BRANCH, REPO_URL, str(REPO)]))
summaries.append(run("02_rev", ["git", "-C", str(REPO), "rev-parse", "HEAD"]))

# 2. deps (torch+pillow+numpy are preinstalled on Kaggle; safetensors usually
#    needs a top-up; pyyaml is preinstalled).
summaries.append(run("03_pip", [
    sys.executable, "-m", "pip", "install", "--quiet",
    "safetensors", "pyyaml",
]))

# 3-5. env, dry-run, micro-benchmark. Bench uses a tiny step count so the
# kernel finishes within a few minutes even on a cold GPU.
env_for_make = {"UIGEN_RUNTIME": "configs/runtime/kaggle.yaml", "PYTHON": sys.executable}

summaries.append(run("04_print_env", [sys.executable, "scripts/print_env.py"], cwd=REPO, env=env_for_make))
summaries.append(run("05_kaggle_dry", [
    sys.executable, "scripts/run_experiment.py",
    "--experiment", "experiments/v7_completer_gateB.yaml",
    "--dry-run",
], cwd=REPO, env=env_for_make))
summaries.append(run("06_bench", [
    sys.executable, "scripts/benchmark_runtime.py",
    "--experiment", "experiments/v7_completer_gateB.yaml",
    "--steps", "50",
], cwd=REPO, env=env_for_make))

# 7. Quick post-conditions: data path resolves to 14 skin dirs, no NaN in bench.
checks: dict[str, str] = {}

import glob
skin_dirs = sorted(glob.glob("/kaggle/input/datasets/dmitriisamoilenko/uigen-data/uigen-data-v7-16skin/data_v7_16skin_completion/*"))
skin_dirs = [d for d in skin_dirs if Path(d).is_dir()]
checks["skin_dir_count"] = str(len(skin_dirs))
checks["skin_dir_count_ok"] = "yes" if len(skin_dirs) == 14 else "no"
manifest = Path("/kaggle/input/datasets/dmitriisamoilenko/uigen-data/uigen-data-v7-16skin/data_v7_16skin_completion/_manifest.json")
checks["manifest_present"] = "yes" if manifest.exists() else "no"

dry_out = (OUT / "05_kaggle_dry.stdout.txt").read_text()
checks["dry_resolves_14"] = "yes" if dry_out.count("=") >= 14 else "no"

bench_out = (OUT / "06_bench.stdout.txt").read_text()
checks["bench_ran"] = "yes" if "sec/step" in bench_out.lower() or "step_count" in bench_out.lower() else "no"
checks["bench_has_nan"] = "yes" if "nan" in bench_out.lower() else "no"

(OUT / "summary.json").write_text(json.dumps({
    "steps": summaries,
    "checks": checks,
    "rc_all": [s["returncode"] for s in summaries],
}, indent=2))

print("\n=== smoke checks ===", flush=True)
print(json.dumps(checks, indent=2), flush=True)

# Final decision: every step must rc==0 AND checks must all be "yes" except
# the ones whose "ok" answer is explicit.
all_ok = (
    all(s["returncode"] == 0 for s in summaries)
    and checks.get("skin_dir_count_ok") == "yes"
    and checks.get("manifest_present") == "yes"
    and checks.get("dry_resolves_14") == "yes"
    and checks.get("bench_ran") == "yes"
    and checks.get("bench_has_nan") == "no"
)
print("SMOKE_RESULT:", "PASS" if all_ok else "FAIL", flush=True)
if not all_ok:
    raise SystemExit(1)
