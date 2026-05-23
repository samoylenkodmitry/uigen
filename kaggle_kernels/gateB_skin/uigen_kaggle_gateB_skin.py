"""V7 Gate B Kaggle training kernel.

Runs the 80 000-step weighted-sampler V7 completer training across the
14 accepted Gate B skins, then evaluates the final checkpoint and writes
a metrics JSON. All outputs land under /kaggle/working/.

Expected runtime on Tesla T4 (from the smoke benchmark):
    ~26 min training + ~1 min eval = ~27 min total.

If the kernel runs out of session time, snapshots every 5 000 steps and
last.safetensors give us a recoverable point to resume from.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_URL = "https://github.com/samoylenkodmitry/uigen.git"
REPO_BRANCH = "main"

WORK = Path("/kaggle/working")
REPO = WORK / "uigen"
RUNS_DIR = WORK / "runs"
OUT = WORK / "gateB_skin_outputs"
OUT.mkdir(parents=True, exist_ok=True)

DATA_ROOT = Path(
    "/kaggle/input/datasets/dmitriisamoilenko/uigen-data/uigen-data-v7-16skin/data_v7_16skin_completion"
)


def run(label: str, cmd: list[str], cwd: Path | None = None, env: dict | None = None,
        capture: bool = True, timeout: int | None = None) -> dict:
    print(f"\n=== {label} ===", flush=True)
    print("$ " + " ".join(cmd), flush=True)
    t0 = time.time()
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    if capture:
        res = subprocess.run(cmd, cwd=cwd, env=full_env, capture_output=True, text=True, timeout=timeout)
        dur = time.time() - t0
        (OUT / f"{label}.stdout.txt").write_text(res.stdout)
        (OUT / f"{label}.stderr.txt").write_text(res.stderr)
        print(res.stdout[-3000:], flush=True)
        if res.returncode != 0:
            print(res.stderr[-3000:], flush=True)
        return {"label": label, "rc": res.returncode, "dur_sec": round(dur, 1)}
    else:
        # Stream live so the kernel's stdout reflects progress in real time.
        res = subprocess.run(cmd, cwd=cwd, env=full_env, timeout=timeout)
        dur = time.time() - t0
        return {"label": label, "rc": res.returncode, "dur_sec": round(dur, 1)}


summaries: list[dict] = []

# 1. Repo + deps.
summaries.append(run("01_clone", ["git", "clone", "--branch", REPO_BRANCH, REPO_URL, str(REPO)]))
summaries.append(run("02_rev", ["git", "-C", str(REPO), "rev-parse", "HEAD"]))
summaries.append(run("03_pip", [sys.executable, "-m", "pip", "install", "--quiet", "safetensors", "pyyaml"]))

# 2. Environment snapshot before training.
env_for_make = {
    "UIGEN_RUNTIME": "configs/runtime/kaggle.yaml",
    "PYTHON": sys.executable,
}
summaries.append(run("04_print_env", [sys.executable, "scripts/print_env.py"], cwd=REPO, env=env_for_make))

# 3. Dry-run, then training. Streamed (capture=False) so we can watch progress
#    in the Kaggle logs.
summaries.append(run("05_kaggle_dry", [
    sys.executable, "scripts/run_experiment.py",
    "--experiment", "experiments/v7_completer_gateB_skin.yaml",
    "--dry-run",
], cwd=REPO, env=env_for_make))

summaries.append(run("06_train", [
    sys.executable, "scripts/run_experiment.py",
    "--experiment", "experiments/v7_completer_gateB_skin.yaml",
], cwd=REPO, env=env_for_make, capture=False))

# 4. Locate the run directory (the experiment YAML pins it).
run_dir = RUNS_DIR / "v7_completer_gateB_skin_16skin"
print(f"\n=== run dir contents ===", flush=True)
for p in sorted(run_dir.glob("*"))[:50]:
    print(f"  {p.name}  ({p.stat().st_size} B)", flush=True)

# 5. Final eval over all 14 skins on last.safetensors.
skin_args = ",".join(
    f"{d.name}={d}" for d in sorted(DATA_ROOT.iterdir()) if d.is_dir()
)
print(f"\nskin_sources count: {skin_args.count('=')}", flush=True)

eval_json = OUT / "gateB_skin_eval.json"
summaries.append(run("07_eval", [
    sys.executable, "scripts/19_eval_v7_completer.py",
    "--state-families", "configs/state_families_classic.yaml",
    "--skin-sources", skin_args,
    "--checkpoint", str(run_dir / "last.safetensors"),
    "--batch", "4",
    "--mask-samples", "4",
    "--device", "cuda",
    "--out-json", str(eval_json),
], cwd=REPO, env=env_for_make))

# 6. Copy the artefacts we want to keep into /kaggle/working/ so they survive
#    the kernel output snapshot.
import shutil
keep = WORK / "kept_artefacts"
keep.mkdir(exist_ok=True)
for name in ("metrics.jsonl", "config.json", "manifest.json",
             "last.safetensors", "best.safetensors"):
    src = run_dir / name
    if src.exists():
        shutil.copy2(src, keep / name)
for snap in run_dir.glob("snapshot_step*.safetensors"):
    shutil.copy2(snap, keep / snap.name)

(OUT / "summary.json").write_text(json.dumps({
    "steps": summaries,
    "eval_json": str(eval_json),
}, indent=2))

# 7. Print top-level eval metrics as the kernel result banner.
if eval_json.exists():
    e = json.loads(eval_json.read_text())
    print("\n=== Gate B aggregate metrics ===", flush=True)
    print(json.dumps(e.get("aggregate", {}), indent=2), flush=True)
    if "per_skin" in e:
        print("\n=== per_skin supported_mae ===", flush=True)
        for sid, m in sorted(e["per_skin"].items()):
            print(f"  {sid:50s}  mae={m['supported_mae']:.5f}  hit5={m['hit5']:.4f}", flush=True)
    if "per_file" in e:
        print("\n=== per_file supported_mae ===", flush=True)
        for fname, m in sorted(e["per_file"].items()):
            print(f"  {fname:18s}  mae={m['supported_mae']:.5f}  hit5={m['hit5']:.4f}", flush=True)

print("\nGATEB_SKIN_RESULT: TRAINED" if all(s["rc"] == 0 for s in summaries) else "GATEB_SKIN_RESULT: FAIL", flush=True)
if not all(s["rc"] == 0 for s in summaries):
    raise SystemExit(1)
