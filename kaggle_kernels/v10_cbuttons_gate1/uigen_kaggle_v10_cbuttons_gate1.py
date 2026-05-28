"""V10 Gate 1 (CBUTTONS.bmp expert) on minimalistic_black_145917e6.

MAIN already passed Gate 1 (MAE 0.000485, hit_5_255 0.998). CBUTTONS is the next
expert per HANDOFF_V10 order: it tests semantic/state-rich fitting (the trainer
sees renders with each transport button pressed; the target is always the full
CBUTTONS.bmp). Only the CBUTTONS expert trains here.

Gate 1 pass = MAE < 0.01, hit_5_255 > 0.90, visually sharp pred-vs-target grid.

Output-dir discipline (same as the MAIN kernel): repo + dataset + run dirs live
under SCRATCH=/tmp; only the small artifact bundle lands in /kaggle/working so
`kaggle kernels output` (no pagination) downloads it in one page.

Demo composition: predicted MAIN (loaded from the uigen-v10-ckpts dataset, the
checkpoint published after the MAIN gate) + predicted CBUTTONS (trained here) +
defaults for the other 9, packaged into a real-Cranamp render side-by-side.
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
OUT = WORK / "v10_cbuttons_outputs"
OUT.mkdir(parents=True, exist_ok=True)

DATA_ROOT = Path(
    "/kaggle/input/datasets/dmitriisamoilenko/uigen-data/uigen-data-v7-16skin/data_v7_16skin_completion"
)
SKIN_DIR = DATA_ROOT / "minimalistic_black_145917e6"
SKIN_ID = "minimalistic_black"
# Trained-checkpoint dataset (published after the MAIN gate). Mount path varies
# (/kaggle/input/<slug> vs /kaggle/input/datasets/<owner>/<slug>), so we search
# all of /kaggle/input for it rather than hardcode a path.
KAGGLE_INPUT = Path("/kaggle/input")
CKPTS_SLUG = "uigen-v10-ckpts"

BMP = "CBUTTONS.bmp"
STEPS = 20000
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

DATA_OUT = SCRATCH / "data_v10_gate1"
RUNS_ROOT = SCRATCH / "runs" / "v10"
RUN_OUT = RUNS_ROOT / "CBUTTONS"


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


def stage_main_checkpoint() -> bool:
    """Copy the published MAIN checkpoint into the runs root so infer_v10 finds
    runs/v10/MAIN/last.safetensors. Searches all mounted input roots (mount
    path is not fixed) for the ckpts dataset, robust to any nesting."""
    roots = [p for p in KAGGLE_INPUT.rglob("*")
             if p.is_dir() and CKPTS_SLUG in p.name]
    roots = roots or [KAGGLE_INPUT]
    cands: list[Path] = []
    for root in roots:
        cands += [p for p in root.rglob("last.safetensors") if p.parent.name == "MAIN"]
    if not cands:  # tar extraction may flatten the MAIN/ prefix
        for root in roots:
            cands += list(root.rglob("MAIN/*.safetensors"))
    if not cands:  # this dataset currently holds only the MAIN checkpoint
        for root in roots:
            cands += list(root.rglob("last.safetensors")) or list(root.rglob("*.safetensors"))
    if not cands:
        print(f"WARN: no MAIN checkpoint found under {KAGGLE_INPUT} ({CKPTS_SLUG}); "
              "demo will default MAIN", flush=True)
        return False
    dst = RUNS_ROOT / "MAIN"
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cands[0], dst / "last.safetensors")
    print(f"staged MAIN checkpoint: {cands[0]} -> {dst / 'last.safetensors'}", flush=True)
    return True


summaries = []
summaries.append(run("01_clone", ["git", "clone", "--branch", REPO_BRANCH, REPO_URL, str(REPO)]))
summaries.append(run("02_rev", ["git", "-C", str(REPO), "rev-parse", "HEAD"]))
summaries.append(run("03_pip", [sys.executable, "-m", "pip", "install", "--quiet",
                                "safetensors", "pyyaml", "Pillow"]))

assert SKIN_DIR.exists(), f"missing skin dir on Kaggle: {SKIN_DIR}"

# 04: generate the Gate 1 dataset. Same generator/scale as MAIN; it writes all
# 11 per-BMP CSVs + targets, so CBUTTONS reuses the identical taxonomy (per-skin
# Kaggle runs are isolated, so the dataset is regenerated here, ~4 min).
summaries.append(run("04_gen_dataset", [
    sys.executable, "scripts/make_v10_bmp_expert_dataset.py",
    "--skin", str(SKIN_DIR), "--skin-id", SKIN_ID,
    "--scale", "gate1", "--out", str(DATA_OUT), "--progress-every", "100",
], cwd=REPO))

csv_path = DATA_OUT / "csv" / "train_CBUTTONS.csv"
target_bmp = DATA_OUT / "targets" / SKIN_ID / BMP
n_rows = sum(1 for _ in csv_path.open()) - 1 if csv_path.exists() else 0
print(f"sanity: csv_rows={n_rows}  target_cbuttons={target_bmp.exists()}", flush=True)

# 05: train CBUTTONS expert (only).
train_cmd = [
    sys.executable, "train_bmp_expert.py",
    "--data", str(DATA_OUT), "--bmp", BMP, "--out", str(RUN_OUT),
    "--steps", str(STEPS), "--batch", str(BATCH), "--lr", str(LR),
    "--base", str(BASE), "--attn-dim", str(ATTN_DIM), "--dec-ch", str(DEC_CH),
    "--heads", str(HEADS), "--attn-layers", str(ATTN_LAYERS),
    "--checkpoint-every", str(CHECKPOINT_EVERY), "--progress-every", str(PROGRESS_EVERY),
    "--num-workers", "2", "--device", "cuda",
]
if AMP:
    train_cmd.append("--amp")
summaries.append(run("05_train_CBUTTONS", train_cmd, cwd=REPO, capture=False))

# 06: eval over the Gate 1 dataset.
eval_dir = RUN_OUT / "eval"
summaries.append(run("06_eval_CBUTTONS", [
    sys.executable, "scripts/eval_bmp_expert.py",
    "--data", str(DATA_OUT), "--bmp", BMP,
    "--checkpoint", str(RUN_OUT / "best.safetensors"),
    "--out", str(eval_dir), "--batch", "4", "--grid-samples", "20", "--device", "cuda",
], cwd=REPO))

# 07: demo with predicted MAIN (staged from ckpt dataset) + CBUTTONS (trained).
staged_main = stage_main_checkpoint()
demo_image = DATA_OUT / "renders" / f"{SKIN_ID}_000000.png"
demo_out = SCRATCH / "v10_cbuttons_demo"
summaries.append(run("07_infer_demo", [
    sys.executable, "infer_v10.py",
    "--image", str(demo_image), "--checkpoints", str(RUNS_ROOT),
    "--out", str(demo_out), "--device", "cuda",
], cwd=REPO))

# Collect headline artifacts into OUT.
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
    "demo_predicted_CBUTTONS.png": demo_out / "predicted_bmps" / "CBUTTONS.png",
    "demo_predicted_MAIN.png": demo_out / "predicted_bmps" / "MAIN.png",
    "demo_experts_used.json": demo_out / "experts_used.json",
}
for name, p in artifacts.items():
    if p.exists():
        shutil.copy2(p, OUT / name)

verdict = {"name": "V10 Gate 1 - CBUTTONS", "skin": SKIN_ID, "steps": STEPS, "batch": BATCH,
           "criteria": "MAE<0.01 AND hit_5_255>0.90", "demo_main_staged": staged_main, "pass": False}
try:
    m = json.loads((eval_dir / "metrics.json").read_text())
    verdict["mae_mean"] = m["mae_mean"]
    verdict["hit_5_255_mean"] = m["hit_5_255_mean"]
    verdict["sobel_mae_mean"] = m["sobel_mae_mean"]
    verdict["pass"] = bool(m.get("gate1_pass", False))
except Exception as e:  # noqa: BLE001
    verdict["error"] = str(e)
(OUT / "summary.json").write_text(json.dumps({"verdict": verdict, "runs": summaries}, indent=2))

print("\n=== V10 CBUTTONS Gate 1 VERDICT ===", flush=True)
print(json.dumps(verdict, indent=2), flush=True)
print("\nDONE.", flush=True)
