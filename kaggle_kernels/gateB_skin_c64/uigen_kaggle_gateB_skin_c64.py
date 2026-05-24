"""V7 Gate B-skin c64 from-scratch training kernel.

Runs experiments/v7_completer_gateB_skin_c64.yaml on T4: c64 base_channels,
skin_embedding_dim=64, state_family-focused mask mix, doubled strip
weights, batch 8 (chosen by c64 batchbench). No --resume-from: c64 is
from scratch.

Expected T4 runtime (from c64 batchbench at batch 8: 0.090 s/step on the
benchmark's uniform file weights; training uses state_family-focused
weights that average ~1.19x more pixels/step):
    80 000 * 0.090 * 1.19 ~= 144 min train
    + per-snapshot eval (~8 snapshots * 3 modes * 5 s) ~= 2 min
    + final per-mode sweep + artefact copy ~= 1 min
    ~= 147 min total.

Snapshots every 10k (8 snapshots over 80k) + last every 5k bound any
session timeout loss to <=5000 steps. --progress-every 200 keeps the
Kaggle log alive every ~18 s, per the CLAUDE.md rule.
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
OUT = WORK / "gateB_skin_c64_outputs"
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
        res = subprocess.run(cmd, cwd=cwd, env=full_env, timeout=timeout)
        dur = time.time() - t0
        return {"label": label, "rc": res.returncode, "dur_sec": round(dur, 1)}


summaries: list[dict] = []
summaries.append(run("01_clone", ["git", "clone", "--branch", REPO_BRANCH, REPO_URL, str(REPO)]))
summaries.append(run("02_rev", ["git", "-C", str(REPO), "rev-parse", "HEAD"]))
summaries.append(run("03_pip", [sys.executable, "-m", "pip", "install", "--quiet", "safetensors", "pyyaml"]))

env_for_make = {"UIGEN_RUNTIME": "configs/runtime/kaggle.yaml", "PYTHON": sys.executable}
summaries.append(run("04_print_env", [sys.executable, "scripts/print_env.py"], cwd=REPO, env=env_for_make))

summaries.append(run("05_kaggle_dry", [
    sys.executable, "scripts/run_experiment.py",
    "--experiment", "experiments/v7_completer_gateB_skin_c64.yaml",
    "--dry-run",
], cwd=REPO, env=env_for_make))

# Streamed (capture=False) so trainer's --progress-every 200 lines
# show up live in the Kaggle log.
summaries.append(run("06_train", [
    sys.executable, "scripts/run_experiment.py",
    "--experiment", "experiments/v7_completer_gateB_skin_c64.yaml",
], cwd=REPO, env=env_for_make, capture=False))

run_dir = RUNS_DIR / "v7_completer_gateB_skin_c64"
print(f"\n=== run dir contents ===", flush=True)
for p in sorted(run_dir.glob("*"))[:50]:
    print(f"  {p.name}  ({p.stat().st_size} B)", flush=True)

skin_args = ",".join(
    f"{d.name}={d}" for d in sorted(DATA_ROOT.iterdir()) if d.is_dir()
)
print(f"\nskin_sources count: {skin_args.count('=')}", flush=True)


def _eval(snap: Path, mode: str, out_json: Path) -> dict:
    if mode == "state_family":
        mask_flags = ["--mask-provenance", "0", "--mask-state-family", "1",
                      "--mask-random-rect", "0", "--mask-whole-file", "0",
                      "--mask-passthrough", "0"]
    elif mode == "random_rect":
        mask_flags = ["--mask-provenance", "0", "--mask-state-family", "0",
                      "--mask-random-rect", "1", "--mask-whole-file", "0",
                      "--mask-passthrough", "0"]
    else:  # "mix" — default trainer mix
        mask_flags = []
    label = f"07_eval_{mode}_{snap.stem}"
    return run(label, [
        sys.executable, "scripts/19_eval_v7_completer.py",
        "--state-families", "configs/state_families_classic.yaml",
        "--skin-sources", skin_args,
        "--checkpoint", str(snap),
        "--batch", "4", "--mask-samples", "4", "--device", "cuda",
        "--out-json", str(out_json),
        *mask_flags,
    ], cwd=REPO, env=env_for_make)


snapshots = sorted(run_dir.glob("snapshot_step*.safetensors"))
last_ckpt = run_dir / "last.safetensors"
if last_ckpt.exists():
    snapshots.append(last_ckpt)

eval_index: list[dict] = []
for snap in snapshots:
    for mode in ("state_family", "random_rect", "mix"):
        out_json = OUT / f"eval_{mode}_{snap.stem}.json"
        s = _eval(snap, mode, out_json)
        summaries.append(s)
        if out_json.exists():
            data = json.loads(out_json.read_text())
            eval_index.append({
                "snapshot": snap.name,
                "mode": mode,
                "aggregate": data.get("aggregate"),
            })

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
    "eval_index": eval_index,
}, indent=2))

print("\n=== per-snapshot per-mode trajectory ===", flush=True)
print(f"{'snapshot':35s} {'mode':14s} {'mae':>9s} {'hit5':>8s}", flush=True)
for row in eval_index:
    agg = row.get("aggregate") or {}
    mae = agg.get("supported_mae", float('nan'))
    hit5 = agg.get("hit5", float('nan'))
    print(f"  {row['snapshot']:33s} {row['mode']:14s} {mae:9.5f} {hit5:8.4f}", flush=True)

all_ok = all(s["rc"] == 0 for s in summaries)
print("\nGATEB_SKIN_C64_RESULT:", "TRAINED" if all_ok else "FAIL", flush=True)
if not all_ok:
    raise SystemExit(1)
