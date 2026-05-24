"""V7 Gate B-BV — BALANCE+VOLUME-only state_family probe on T4.

Runs experiments/v7_completer_gateB_bv.yaml: c48, batch 12,
within-file-replacement=false, state_family-only, file-sampling-weights-
mode=REPLACE so the trainer sees only BALANCE.bmp and VOLUME.bmp.

After training, evals each snapshot per-mode/per-file/per-skin and dumps
visual diffs (target / pred / |diff|*5) for the 3 worst skins on each
of the two files, using scripts/dump_v7_completer_diffs.py. Diffs let us
see *how* the model fails (smear, hue drift, missing thumb, wrong frame,
hallucinated content) — scalar metrics alone don't.

Expected T4 runtime:
    BALANCE/VOLUME are the slowest files (vertical-strip eval) but the
    only thing the trainer sees, so the per-step cost is bounded by
    those tensor sizes. Empirically the strip kernel hit ~0.20 s/step
    with batch 12 across mixed files. BV-only should be in that range:
    30 000 * 0.20 ~= 100 min train
    + 6 snapshots * 2 modes * 5 s ~= 1 min
    + diff dump for ~6 (skin, file) tuples * 3 seeds ~= 30 s
    ~= 102 min total.

Snapshots every 5k bound any forced-cancel loss to <= 5000 steps.
Trainer prints by_mode / by_file / by_skin breakdowns each progress
window, plus a "effective file probabilities" line at start that should
show BALANCE=0.50, VOLUME=0.50 and nothing else.
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
OUT = WORK / "gateB_bv_outputs"
OUT.mkdir(parents=True, exist_ok=True)

DATA_ROOT = Path(
    "/kaggle/input/datasets/dmitriisamoilenko/uigen-data/uigen-data-v7-16skin/data_v7_16skin_completion"
)

BV_FILES = ("BALANCE.bmp", "VOLUME.bmp")
NUM_WORST_SKINS = 3


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
summaries.append(run("03_pip", [sys.executable, "-m", "pip", "install", "--quiet",
                                "safetensors", "pyyaml", "Pillow"]))

env_for_make = {"UIGEN_RUNTIME": "configs/runtime/kaggle.yaml", "PYTHON": sys.executable}
summaries.append(run("04_print_env", [sys.executable, "scripts/print_env.py"], cwd=REPO, env=env_for_make))

summaries.append(run("05_kaggle_dry", [
    sys.executable, "scripts/run_experiment.py",
    "--experiment", "experiments/v7_completer_gateB_bv.yaml",
    "--dry-run",
], cwd=REPO, env=env_for_make))

# Streamed (capture=False) so trainer's progress + per-mode/per-file/per-skin
# breakdowns + effective-file-probabilities line show up live in the Kaggle log.
summaries.append(run("06_train", [
    sys.executable, "scripts/run_experiment.py",
    "--experiment", "experiments/v7_completer_gateB_bv.yaml",
], cwd=REPO, env=env_for_make, capture=False))

run_dir = RUNS_DIR / "v7_completer_gateB_bv"
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


def _worst_skins_for_file(per_file_skin: dict, file_name: str, n: int) -> list[str]:
    """Given a dict {skin: {file: {supported_mae, ...}}}, return the top-n
    skin ids by supported_mae on `file_name`. Fall back to global per-skin
    if the per-file-per-skin breakdown is unavailable."""
    rows: list[tuple[str, float]] = []
    for skin, files in per_file_skin.items():
        if not isinstance(files, dict):
            continue
        sub = files.get(file_name)
        if not isinstance(sub, dict):
            continue
        m = sub.get("supported_mae")
        if m is not None:
            rows.append((skin, float(m)))
    rows.sort(key=lambda kv: -kv[1])
    return [s for s, _ in rows[:n]]


# Pick the per-snapshot state_family eval at the LAST checkpoint to drive
# the diff dump. We look at per_skin first (eval script reports per-skin
# aggregated across all files); if a per-file-per-skin breakdown is
# present we use that for more targeted picks.
last_sf = next(
    (r for r in reversed(eval_index)
     if r["mode"] == "state_family" and r["snapshot"] == "last.safetensors"),
    None,
)
worst_per_file: dict[str, list[str]] = {}
if last_sf is not None:
    per_skin = last_sf.get("per_skin") or {}
    if isinstance(per_skin, dict):
        items = sorted(
            ((sid, v.get("supported_mae")) for sid, v in per_skin.items()
             if isinstance(v, dict) and v.get("supported_mae") is not None),
            key=lambda kv: -float(kv[1]),
        )
        global_worst = [s for s, _ in items[:NUM_WORST_SKINS]]
        for fn in BV_FILES:
            worst_per_file[fn] = global_worst
print("\n=== worst skins per BV file ===", flush=True)
print(json.dumps(worst_per_file, indent=2), flush=True)

# Dump diffs for the worst-3 union per file. Use last.safetensors as the
# checkpoint (final-trained model).
diff_dir = OUT / "diffs"
diff_dir.mkdir(exist_ok=True)
all_worst_skins = sorted({s for skins in worst_per_file.values() for s in skins})
if all_worst_skins:
    summaries.append(run("08_dump_diffs", [
        sys.executable, "scripts/dump_v7_completer_diffs.py",
        "--state-families", "configs/state_families_classic.yaml",
        "--skin-sources", skin_args,
        "--checkpoint", str(last_ckpt),
        "--files", ",".join(BV_FILES),
        "--skins", ",".join(all_worst_skins),
        "--mask-mode", "state_family",
        "--num-seeds", "3",
        "--device", "cuda",
        "--out-dir", str(diff_dir),
    ], cwd=REPO, env=env_for_make))
else:
    print("(no worst-skin list resolved; skipping diff dump)", flush=True)

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
    "worst_per_file": worst_per_file,
}, indent=2))

print("\n=== per-snapshot per-mode state_family/mix aggregate ===", flush=True)
print(f"{'snapshot':35s} {'mode':14s} {'mae':>9s} {'hit5':>8s}", flush=True)
for row in eval_index:
    agg = row.get("aggregate") or {}
    mae = agg.get("supported_mae", float('nan'))
    hit5 = agg.get("hit5", float('nan'))
    print(f"  {row['snapshot']:33s} {row['mode']:14s} {mae:9.5f} {hit5:8.4f}", flush=True)

# Headline per_file table for state_family at the last snapshot.
if last_sf and last_sf.get("per_file"):
    print("\n=== last state_family per_file ===", flush=True)
    pf = last_sf["per_file"]
    items = list(pf.items()) if isinstance(pf, dict) else [(r.get("file", "?"), r) for r in pf]
    items.sort(key=lambda kv: -(kv[1].get("supported_mae", 0) if isinstance(kv[1], dict) else 0))
    for k, v in items:
        if not isinstance(v, dict): continue
        print(f"  {k:30s} mae={v.get('supported_mae'):.5f}  hit5={v.get('hit5'):.4f}", flush=True)

if last_sf and last_sf.get("per_skin"):
    print("\n=== last state_family per_skin (top 6 worst) ===", flush=True)
    ps = last_sf["per_skin"]
    items = list(ps.items()) if isinstance(ps, dict) else [(r.get("skin", "?"), r) for r in ps]
    items.sort(key=lambda kv: -(kv[1].get("supported_mae", 0) if isinstance(kv[1], dict) else 0))
    for k, v in items[:6]:
        if not isinstance(v, dict): continue
        print(f"  {k:40s} mae={v.get('supported_mae'):.5f}  hit5={v.get('hit5'):.4f}", flush=True)

all_ok = all(s["rc"] == 0 for s in summaries)
print("\nGATEB_BV_RESULT:", "TRAINED" if all_ok else "FAIL", flush=True)
if not all_ok:
    raise SystemExit(1)
