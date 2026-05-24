"""V7 Gate B-strip — state_family-only training on the failing strip files.

Runs experiments/v7_completer_gateB_strip.yaml on T4: c48, batch 12,
within-file-replacement=false, state_family mask only, strip-only file
weights (BALANCE / VOLUME / POSBAR / PLAYPAUS / MONOSTER / CBUTTONS).

Trainer prints per-mode / per-file / per-skin breakdowns each progress
window (every 200 steps, ~30 s on T4), so the live log surfaces which
strip file and which skin is dragging the mean. We also eval each
snapshot for state_family and mix mode to track the curve against c48
/ c64 16-skin Gate B-skin runs.

Expected T4 runtime: 40 000 steps * ~0.07 s/step (c48, batch 12,
state_family masks which are cheaper than provenance) ~= 47 min train
+ 8 snapshots * 2 modes * 5 s eval ~= 1 min
+ final summary ~= 1 min
~= 50 min total.

Snapshots every 5k + last every 5k bound any session-timeout loss to
<= 5000 steps.
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
OUT = WORK / "gateB_strip_outputs"
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
    "--experiment", "experiments/v7_completer_gateB_strip.yaml",
    "--dry-run",
], cwd=REPO, env=env_for_make))

# Streamed (capture=False) so trainer's progress + per-mode/per-file/per-skin
# breakdowns show up live in the Kaggle log.
summaries.append(run("06_train", [
    sys.executable, "scripts/run_experiment.py",
    "--experiment", "experiments/v7_completer_gateB_strip.yaml",
], cwd=REPO, env=env_for_make, capture=False))

run_dir = RUNS_DIR / "v7_completer_gateB_strip"
print(f"\n=== run dir contents ===", flush=True)
for p in sorted(run_dir.glob("*"))[:50]:
    print(f"  {p.name}  ({p.stat().st_size} B)", flush=True)

skin_args = ",".join(
    f"{d.name}={d}" for d in sorted(DATA_ROOT.iterdir()) if d.is_dir()
)
print(f"\nskin_sources count: {skin_args.count('=')}", flush=True)


def _eval(snap: Path, mode: str, out_json: Path) -> dict:
    # state_family-only and mix; random_rect is already at mae 0.011 across
    # both c48 and c64 so retraining strip files won't move it — skip.
    if mode == "state_family":
        mask_flags = ["--mask-provenance", "0", "--mask-state-family", "1",
                      "--mask-random-rect", "0", "--mask-whole-file", "0",
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
    for mode in ("state_family", "mix"):
        out_json = OUT / f"eval_{mode}_{snap.stem}.json"
        s = _eval(snap, mode, out_json)
        summaries.append(s)
        if out_json.exists():
            data = json.loads(out_json.read_text())
            eval_index.append({
                "snapshot": snap.name,
                "mode": mode,
                "aggregate": data.get("aggregate"),
                "per_file": data.get("per_file"),
                "per_skin": data.get("per_skin"),
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

print("\n=== per-snapshot state_family aggregate ===", flush=True)
print(f"{'snapshot':35s} {'mode':14s} {'mae':>9s} {'hit5':>8s}", flush=True)
for row in eval_index:
    agg = row.get("aggregate") or {}
    mae = agg.get("supported_mae", float('nan'))
    hit5 = agg.get("hit5", float('nan'))
    print(f"  {row['snapshot']:33s} {row['mode']:14s} {mae:9.5f} {hit5:8.4f}", flush=True)

# Last-snapshot per-file table (state_family) is the headline for this probe.
last_sf = next((r for r in reversed(eval_index)
                if r["mode"] == "state_family" and r["snapshot"] == "last.safetensors"),
               None)
if last_sf and last_sf.get("per_file"):
    print("\n=== last state_family per_file ===", flush=True)
    pf = last_sf["per_file"]
    items = list(pf.items()) if isinstance(pf, dict) else [(r.get("file", "?"), r) for r in pf]
    items.sort(key=lambda kv: -(kv[1].get("supported_mae", 0) if isinstance(kv[1], dict) else 0))
    for k, v in items:
        if not isinstance(v, dict): continue
        print(f"  {k:30s} mae={v.get('supported_mae'):.5f}  hit5={v.get('hit5'):.4f}", flush=True)

all_ok = all(s["rc"] == 0 for s in summaries)
print("\nGATEB_STRIP_RESULT:", "TRAINED" if all_ok else "FAIL", flush=True)
if not all_ok:
    raise SystemExit(1)
