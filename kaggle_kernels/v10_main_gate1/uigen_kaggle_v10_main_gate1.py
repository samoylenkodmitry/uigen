"""V10 Gate 1 (MAIN.bmp expert) on minimalistic_black_145917e6.

Per HANDOFF_V10_BMP_EXPERTS.md: only the MAIN expert trains here. Other experts
are NOT launched. Gate 1 pass = MAE < 0.01, hit_5_255 > 0.90, visually sharp on
the one-skin overfit. Failing means stop and debug, not move to CBUTTONS.

Steps:

    01_clone repo (main)
    02_rev / 03_pip
    04_gen Gate 1 dataset (deterministic state sweeps + random geometry)
    05_train MAIN expert (full dims from HANDOFF_V10 defaults)
    06_eval (MAE / hit_5_255 / Sobel + pred-vs-target grid)
    07_package skin.wsz (predicted MAIN + defaults for the rest)
    08_real-Cranamp render side-by-side from the packaged skin

All required Gate 1 artifacts (checkpoint, train/eval log, pred-vs-target grid,
metrics, packaged skin, Cranamp render side-by-side) land under
/kaggle/working/v10_main_outputs.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REPO_URL = "https://github.com/samoylenkodmitry/uigen.git"
REPO_BRANCH = "main"

WORK = Path("/kaggle/working")
REPO = WORK / "uigen"
OUT = WORK / "v10_main_outputs"
OUT.mkdir(parents=True, exist_ok=True)

# Existing Kaggle dataset (same as prior V7 kernels).
DATA_ROOT = Path(
    "/kaggle/input/datasets/dmitriisamoilenko/uigen-data/uigen-data-v7-16skin/data_v7_16skin_completion"
)
SKIN_DIR = DATA_ROOT / "minimalistic_black_145917e6"
SKIN_ID = "minimalistic_black"

# Training config (HANDOFF_V10 starting defaults).
STEPS = 20000
BATCH = 8
LR = 3e-4
BASE = 48
ATTN_DIM = 256
DEC_CH = 128
HEADS = 4
ATTN_LAYERS = 2
CHECKPOINT_EVERY = 2000
PROGRESS_EVERY = 100

DATA_OUT = WORK / "data_v10_gate1"
RUN_OUT = WORK / "runs" / "v10" / "MAIN"


def run(label, cmd, cwd=None, capture=True):
    print(f"\n=== {label} ===", flush=True)
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    t0 = time.time()
    if capture:
        res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        (OUT / f"{label}.stdout.txt").write_text(res.stdout)
        (OUT / f"{label}.stderr.txt").write_text(res.stderr)
        print(res.stdout[-3000:], flush=True)
        if res.returncode != 0:
            print(res.stderr[-3000:], flush=True)
        return {"label": label, "rc": res.returncode, "dur_sec": round(time.time() - t0, 1)}
    res = subprocess.run(cmd, cwd=cwd)
    return {"label": label, "rc": res.returncode, "dur_sec": round(time.time() - t0, 1)}


summaries = []
summaries.append(run("01_clone", ["git", "clone", "--branch", REPO_BRANCH, REPO_URL, str(REPO)]))
summaries.append(run("02_rev", ["git", "-C", str(REPO), "rev-parse", "HEAD"]))
summaries.append(run("03_pip", [sys.executable, "-m", "pip", "install", "--quiet",
                                "safetensors", "pyyaml", "Pillow"]))

assert SKIN_DIR.exists(), f"missing skin dir on Kaggle: {SKIN_DIR}"

# 04: generate Gate 1 dataset (~743 variants, ~3-4 minutes CPU).
summaries.append(run("04_gen_dataset", [
    sys.executable, "scripts/make_v10_bmp_expert_dataset.py",
    "--skin", str(SKIN_DIR), "--skin-id", SKIN_ID,
    "--scale", "gate1", "--out", str(DATA_OUT),
    "--progress-every", "100",
], cwd=REPO))

# Sanity-check artifact: row count + target size + gaps.
csv_path = DATA_OUT / "csv" / "train_MAIN.csv"
target_main = DATA_OUT / "targets" / SKIN_ID / "MAIN.bmp"
n_rows = sum(1 for _ in csv_path.open()) - 1 if csv_path.exists() else 0
print(f"sanity: csv_rows={n_rows}  target_main={target_main.exists()}  "
      f"gaps={(DATA_OUT / 'renderer_gaps.json').exists()}", flush=True)

# 05: train MAIN expert (only).
summaries.append(run("05_train_MAIN", [
    sys.executable, "train_bmp_expert.py",
    "--data", str(DATA_OUT), "--bmp", "MAIN.bmp", "--out", str(RUN_OUT),
    "--steps", str(STEPS), "--batch", str(BATCH), "--lr", str(LR),
    "--base", str(BASE), "--attn-dim", str(ATTN_DIM), "--dec-ch", str(DEC_CH),
    "--heads", str(HEADS), "--attn-layers", str(ATTN_LAYERS),
    "--checkpoint-every", str(CHECKPOINT_EVERY), "--progress-every", str(PROGRESS_EVERY),
    "--num-workers", "2", "--device", "cuda",
], cwd=REPO, capture=False))

# 06: eval over the Gate 1 dataset (uses best.safetensors).
eval_dir = RUN_OUT / "eval"
summaries.append(run("06_eval_MAIN", [
    sys.executable, "scripts/eval_bmp_expert.py",
    "--data", str(DATA_OUT), "--bmp", "MAIN.bmp",
    "--checkpoint", str(RUN_OUT / "best.safetensors"),
    "--out", str(eval_dir), "--batch", "4", "--grid-samples", "20",
    "--device", "cuda",
], cwd=REPO))

# 07: package skin.wsz from a representative dataset render + predicted MAIN.
# Use the first dataset render as the input image so the demo is closed-loop:
# render-from-source -> predicted MAIN.bmp -> defaults for other 10 -> .wsz.
demo_image = DATA_OUT / "renders" / f"{SKIN_ID}_000000.png"
demo_out = WORK / "v10_main_demo"
summaries.append(run("07_infer_demo", [
    sys.executable, "infer_v10.py",
    "--image", str(demo_image),
    "--checkpoints", str(RUN_OUT.parent),
    "--out", str(demo_out),
    "--device", "cuda",
], cwd=REPO))

# Collect headline artifacts into OUT for easy download.
import shutil
artifacts = {
    "best.safetensors": RUN_OUT / "best.safetensors",
    "last.safetensors": RUN_OUT / "last.safetensors",
    "config.json": RUN_OUT / "config.json",
    "metrics.jsonl": RUN_OUT / "metrics.jsonl",
    "eval_metrics.json": eval_dir / "metrics.json",
    "eval_per_variant.csv": eval_dir / "per_variant.csv",
    "pred_vs_target_grid.png": eval_dir / "pred_vs_target_grid.png",
    "demo_side_by_side.png": demo_out / "side_by_side.png",
    "demo_skin.wsz": demo_out / "skin" / "skin.wsz",
    "demo_normalized.png": demo_out / "normalized.png",
    "demo_render_cranamp.png": demo_out / "render_cranamp.png",
    "demo_predicted_MAIN.png": demo_out / "predicted_bmps" / "MAIN.png",
}
for name, p in artifacts.items():
    if p.exists():
        shutil.copy2(p, OUT / name)

# Gate 1 verdict (eval metrics).
verdict = {"name": "V10 Gate 1 - MAIN", "skin": SKIN_ID, "steps": STEPS, "batch": BATCH,
           "criteria": "MAE<0.01 AND hit_5_255>0.90", "pass": False}
try:
    m = json.loads((eval_dir / "metrics.json").read_text())
    verdict["mae_mean"] = m["mae_mean"]
    verdict["hit_5_255_mean"] = m["hit_5_255_mean"]
    verdict["sobel_mae_mean"] = m["sobel_mae_mean"]
    verdict["pass"] = bool(m.get("gate1_pass", False))
except Exception as e:  # noqa: BLE001
    verdict["error"] = str(e)
(OUT / "summary.json").write_text(json.dumps(
    {"verdict": verdict, "runs": summaries}, indent=2))

print("\n=== V10 MAIN Gate 1 VERDICT ===", flush=True)
print(json.dumps(verdict, indent=2), flush=True)
print("\nDONE.", flush=True)
