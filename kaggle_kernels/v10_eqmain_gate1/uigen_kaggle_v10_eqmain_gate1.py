"""V10 Gate 1 (EQMAIN.bmp expert) on minimalistic_black_145917e6.

MAIN and CBUTTONS passed Gate 1. EQMAIN is next per HANDOFF_V10 order: it tests
state-rich output from partial visual cues (renders sweep each EQ band across
positions, random curves, on/off/auto; target is always the full EQMAIN.bmp).
Only the EQMAIN expert trains here. First gate run to START with early-stop.

Gate 1 pass = MAE < 0.01, hit_5_255 > 0.90, visually sharp pred-vs-target grid.

Conventions (see the MAIN/CBUTTONS kernels): repo+dataset+run dirs under
SCRATCH=/tmp, only the small bundle in /kaggle/working (single-page download);
early-stop bounded by STEPS; demo composes ALL experts published in the
uigen-v10-ckpts dataset (MAIN + CBUTTONS) plus the freshly-trained EQMAIN, with
defaults for the rest.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_URL = "https://github.com/samoylenkodmitry/uigen.git"
REPO_BRANCH = "main"

SCRATCH = Path("/tmp/v10_work")
SCRATCH.mkdir(parents=True, exist_ok=True)
REPO = SCRATCH / "uigen"
WORK = Path("/kaggle/working")
OUT = WORK / "v10_eqmain_outputs"
OUT.mkdir(parents=True, exist_ok=True)

DATA_ROOT = Path(
    "/kaggle/input/datasets/dmitriisamoilenko/uigen-data/uigen-data-v7-16skin/data_v7_16skin_completion"
)
SKIN_DIR = DATA_ROOT / "minimalistic_black_145917e6"
SKIN_ID = "minimalistic_black"
KAGGLE_INPUT = Path("/kaggle/input")
CKPTS_SLUG = "uigen-v10-ckpts"

BMP = "EQMAIN.bmp"
STEPS = 15000
BATCH = 4
LR = 3e-4
BASE = 48
ATTN_DIM = 256
DEC_CH = 128
HEADS = 4
ATTN_LAYERS = 2
CHECKPOINT_EVERY = 2000
PROGRESS_EVERY = 100
AMP = True
EVAL_EVERY = 500
EVAL_MAX_ITEMS = 256
EARLY_STOP_MAE = 0.008
EARLY_STOP_HIT5 = 0.93
EARLY_STOP_PATIENCE = 2

DATA_OUT = SCRATCH / "data_v10_gate1"
RUNS_ROOT = SCRATCH / "runs" / "v10"
RUN_OUT = RUNS_ROOT / "EQMAIN"


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


def stage_prior_checkpoints() -> list[str]:
    """Copy every prior expert checkpoint from the uigen-v10-ckpts dataset into
    the runs root so infer_v10 composes them in the demo. Returns staged stems.
    Mount path is not fixed, so search all /kaggle/input roots."""
    roots = [p for p in KAGGLE_INPUT.rglob("*") if p.is_dir() and CKPTS_SLUG in p.name] or [KAGGLE_INPUT]
    staged: list[str] = []
    for stem in ("MAIN", "CBUTTONS"):
        cands = []
        for root in roots:
            cands += [p for p in root.rglob("last.safetensors") if p.parent.name == stem]
        if not cands:  # tar may flatten; fall back to any <stem>/*.safetensors
            for root in roots:
                cands += list(root.rglob(f"{stem}/*.safetensors"))
        if cands:
            dst = RUNS_ROOT / stem
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cands[0], dst / "last.safetensors")
            staged.append(stem)
    print(f"staged prior checkpoints: {staged or '(none)'}", flush=True)
    return staged


summaries = []
summaries.append(run("01_clone", ["git", "clone", "--branch", REPO_BRANCH, REPO_URL, str(REPO)]))
summaries.append(run("02_rev", ["git", "-C", str(REPO), "rev-parse", "HEAD"]))
summaries.append(run("03_pip", [sys.executable, "-m", "pip", "install", "--quiet",
                                "safetensors", "pyyaml", "Pillow"]))

assert SKIN_DIR.exists(), f"missing skin dir on Kaggle: {SKIN_DIR}"

summaries.append(run("04_gen_dataset", [
    sys.executable, "scripts/make_v10_bmp_expert_dataset.py",
    "--skin", str(SKIN_DIR), "--skin-id", SKIN_ID,
    "--scale", "gate1", "--out", str(DATA_OUT), "--progress-every", "100",
], cwd=REPO))

csv_path = DATA_OUT / "csv" / "train_EQMAIN.csv"
n_rows = sum(1 for _ in csv_path.open()) - 1 if csv_path.exists() else 0
print(f"sanity: csv_rows={n_rows} target={(DATA_OUT / 'targets' / SKIN_ID / BMP).exists()}", flush=True)

train_cmd = [
    sys.executable, "train_bmp_expert.py",
    "--data", str(DATA_OUT), "--bmp", BMP, "--out", str(RUN_OUT),
    "--steps", str(STEPS), "--batch", str(BATCH), "--lr", str(LR),
    "--base", str(BASE), "--attn-dim", str(ATTN_DIM), "--dec-ch", str(DEC_CH),
    "--heads", str(HEADS), "--attn-layers", str(ATTN_LAYERS),
    "--checkpoint-every", str(CHECKPOINT_EVERY), "--progress-every", str(PROGRESS_EVERY),
    "--eval-every", str(EVAL_EVERY), "--eval-max-items", str(EVAL_MAX_ITEMS),
    "--early-stop", "--early-stop-mae", str(EARLY_STOP_MAE),
    "--early-stop-hit5", str(EARLY_STOP_HIT5), "--early-stop-patience", str(EARLY_STOP_PATIENCE),
    "--num-workers", "2", "--device", "cuda",
]
if AMP:
    train_cmd.append("--amp")
summaries.append(run("05_train_EQMAIN", train_cmd, cwd=REPO, capture=False))

eval_dir = RUN_OUT / "eval"
summaries.append(run("06_eval_EQMAIN", [
    sys.executable, "scripts/eval_bmp_expert.py",
    "--data", str(DATA_OUT), "--bmp", BMP,
    "--checkpoint", str(RUN_OUT / "best.safetensors"),
    "--out", str(eval_dir), "--batch", "4", "--grid-samples", "20", "--device", "cuda",
], cwd=REPO))

staged = stage_prior_checkpoints()
demo_image = DATA_OUT / "renders" / f"{SKIN_ID}_000000.png"
demo_out = SCRATCH / "v10_eqmain_demo"
summaries.append(run("07_infer_demo", [
    sys.executable, "infer_v10.py",
    "--image", str(demo_image), "--checkpoints", str(RUNS_ROOT),
    "--out", str(demo_out), "--device", "cuda",
], cwd=REPO))

artifacts = {
    "best.safetensors": RUN_OUT / "best.safetensors",
    "last.safetensors": RUN_OUT / "last.safetensors",
    "config.json": RUN_OUT / "config.json",
    "metrics.jsonl": RUN_OUT / "metrics.jsonl",
    "eval_progress.jsonl": RUN_OUT / "eval_progress.jsonl",
    "eval_metrics.json": eval_dir / "metrics.json",
    "eval_per_variant.csv": eval_dir / "per_variant.csv",
    "pred_vs_target_grid.png": eval_dir / "pred_vs_target_grid.png",
    "demo_side_by_side.png": demo_out / "side_by_side.png",
    "demo_skin.wsz": demo_out / "skin" / "skin.wsz",
    "demo_render_cranamp.png": demo_out / "render_cranamp.png",
    "demo_predicted_EQMAIN.png": demo_out / "predicted_bmps" / "EQMAIN.png",
    "demo_experts_used.json": demo_out / "experts_used.json",
}
for name, p in artifacts.items():
    if p.exists():
        shutil.copy2(p, OUT / name)

verdict = {"name": "V10 Gate 1 - EQMAIN", "skin": SKIN_ID, "steps_cap": STEPS, "batch": BATCH,
           "criteria": "MAE<0.01 AND hit_5_255>0.90", "demo_prior_staged": staged, "pass": False}
try:
    m = json.loads((eval_dir / "metrics.json").read_text())
    verdict["mae_mean"] = m["mae_mean"]
    verdict["hit_5_255_mean"] = m["hit_5_255_mean"]
    verdict["sobel_mae_mean"] = m["sobel_mae_mean"]
    verdict["pass"] = bool(m.get("gate1_pass", False))
except Exception as e:  # noqa: BLE001
    verdict["error"] = str(e)
(OUT / "summary.json").write_text(json.dumps({"verdict": verdict, "runs": summaries}, indent=2))

print("\n=== V10 EQMAIN Gate 1 VERDICT ===", flush=True)
print(json.dumps(verdict, indent=2), flush=True)
print("\nDONE.", flush=True)
