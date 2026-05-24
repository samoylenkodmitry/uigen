"""V7 Gate B-skin state_family-focused continuation on Kaggle.

Resumes from the 75k Gate B-skin continuation checkpoint and trains
another 30 000 steps with the easy mask paths (whole_file, passthrough)
zeroed and slider-strip file weights doubled. See the experiment YAML
for the full rationale.

Expected T4 runtime (per-batch cost is ~1.19x batchbench-uniform under
the focused weights, ~0.13 s/step at batch 14 -> 30k * 0.156 ~ 78 min
train + 6 snapshot evals (state_family + default mix each) ~ 5 min
plus a final per-mode sweep at end.

Attached datasets:
    uigen-data                       -- the 16-skin completion dataset.
    uigen-gatebskin-cont-75k-ckpt    -- the 75k resume seed.
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
OUT = WORK / "gateB_skin_sf_outputs"
OUT.mkdir(parents=True, exist_ok=True)

DATA_ROOT = Path(
    "/kaggle/input/datasets/dmitriisamoilenko/uigen-data/uigen-data-v7-16skin/data_v7_16skin_completion"
)
CKPT = Path(
    "/kaggle/input/datasets/dmitriisamoilenko/uigen-gatebskin-cont-75k-ckpt/last.safetensors"
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

# Sanity: the resume checkpoint must be mounted.
print(f"\n=== 00_checkpoint_present ===", flush=True)
print(f"CKPT = {CKPT}", flush=True)
print(f"exists = {CKPT.exists()}", flush=True)
if not CKPT.exists():
    print("FATAL: source checkpoint not mounted; cannot resume.", flush=True)
    raise SystemExit(2)
print(f"size = {CKPT.stat().st_size} B", flush=True)

summaries.append(run("01_clone", ["git", "clone", "--branch", REPO_BRANCH, REPO_URL, str(REPO)]))
summaries.append(run("02_rev", ["git", "-C", str(REPO), "rev-parse", "HEAD"]))
summaries.append(run("03_pip", [sys.executable, "-m", "pip", "install", "--quiet", "safetensors", "pyyaml"]))

env_for_make = {"UIGEN_RUNTIME": "configs/runtime/kaggle.yaml", "PYTHON": sys.executable}
summaries.append(run("04_print_env", [sys.executable, "scripts/print_env.py"], cwd=REPO, env=env_for_make))

summaries.append(run("05_kaggle_dry", [
    sys.executable, "scripts/run_experiment.py",
    "--experiment", "experiments/v7_completer_gateB_skin_statefamily.yaml",
    "--dry-run",
], cwd=REPO, env=env_for_make))

# Streamed (capture=False) so the in-trainer --progress-every 200 lines
# show up live in the Kaggle log.
summaries.append(run("06_train", [
    sys.executable, "scripts/run_experiment.py",
    "--experiment", "experiments/v7_completer_gateB_skin_statefamily.yaml",
], cwd=REPO, env=env_for_make, capture=False))

run_dir = RUNS_DIR / "v7_completer_gateB_skin_statefamily"
print(f"\n=== run dir contents ===", flush=True)
for p in sorted(run_dir.glob("*"))[:50]:
    print(f"  {p.name}  ({p.stat().st_size} B)", flush=True)

skin_args = ",".join(
    f"{d.name}={d}" for d in sorted(DATA_ROOT.iterdir()) if d.is_dir()
)
print(f"\nskin_sources count: {skin_args.count('=')}", flush=True)


# Per-snapshot mode-conditioned eval. We re-run eval against every snapshot
# in both state_family-only and the default mix, so the kernel output gives
# us the per-mode trajectory without needing inline-eval in the trainer.
def _eval_label(snap: Path, mode: str) -> str:
    return f"07_eval_{mode}_{snap.stem}"


def _eval(snap: Path, mode: str, out_json: Path) -> dict:
    if mode == "state_family":
        mask_flags = [
            "--mask-provenance", "0",
            "--mask-state-family", "1",
            "--mask-random-rect", "0",
            "--mask-whole-file", "0",
            "--mask-passthrough", "0",
        ]
    elif mode == "random_rect":
        mask_flags = [
            "--mask-provenance", "0",
            "--mask-state-family", "0",
            "--mask-random-rect", "1",
            "--mask-whole-file", "0",
            "--mask-passthrough", "0",
        ]
    else:  # "mix" = default
        mask_flags = []
    return run(_eval_label(snap, mode), [
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

# Persist artefacts: every snapshot + last + best + metrics + config + manifest.
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
print(f"{'snapshot':35s} {'mode':14s} {'mae':>8s} {'hit5':>8s}", flush=True)
for row in eval_index:
    agg = row.get("aggregate") or {}
    mae = agg.get("supported_mae", float('nan'))
    hit5 = agg.get("hit5", float('nan'))
    print(f"  {row['snapshot']:33s} {row['mode']:14s} {mae:8.5f} {hit5:8.4f}", flush=True)

all_ok = all(s["rc"] == 0 for s in summaries)
print("\nGATEB_SKIN_SF_RESULT:", "TRAINED" if all_ok else "FAIL", flush=True)
if not all_ok:
    raise SystemExit(1)
