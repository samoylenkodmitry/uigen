"""V10 Gate 2 — the 11 BMP experts across 14 skins (in-distribution generalization).

Gate 1 (one-skin overfit on minimalistic_black) PASSED for all 11 experts. Gate 2
is the real generalization test: ONE expert per BMP must reproduce the correct,
SKIN-SPECIFIC target by reading the skin's look from the full Cranamp render. No
architecture change vs Gate 1 — different skin -> different render -> different
target BMP. The 14-skin dataset is generated in-kernel (per-skin deterministic
state sweeps, --append into one out dir).

Per-expert recipe (the validated triage from Gate 1):
  1. L1 train (AMP, base config, progressive decoder, early-stop, capped).
     AMP batch 4 is confirmed-safe on T4 (Gate-1 MAIN/CBUTTONS); FP32 batch 4
     OOMs the local 8GB card and is tight on T4. TITLEBAR's Gate-1 "freeze"
     happened under BOTH AMP and FP32 (it is the constant-target degenerate
     local-min, not an AMP artifact) and is cured by the adversarial stage
     regardless of precision, so AMP here costs nothing.
  2. eval across all 14 skins -> gate2_pass = EVERY skin clears mae<0.01 &
     hit5>0.90 (a mean can pass while one skin fails, so the gate is per-skin).
  3. If gate2 fails, ADVERSARIAL fine-tune from the L1 anchor (the recipe that
     crossed EQMAIN/BALANCE/TITLEBAR in Gate 1) and re-eval; keep the better ckpt.

Conventions (see the Gate-1 kernels): repo+dataset+run dirs under SCRATCH=/tmp,
only the small bundle in /kaggle/working (single-page download). A global
wall-clock budget stages whatever finished and exits cleanly before Kaggle's 12h
kill, so partial output (and checkpoints) always persist. EXPERTS is ordered
easy->hard and can be trimmed to run Gate 2 in waves across sessions.
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
OUT = WORK / "v10_gate2_outputs"
OUT.mkdir(parents=True, exist_ok=True)

DATA_ROOT = Path(
    "/kaggle/input/datasets/dmitriisamoilenko/uigen-data/uigen-data-v7-16skin/data_v7_16skin_completion"
)

# The 14 skins materialized in the v7 16-skin completion set. (skin_dir, skin_id)
SKINS = [
    ("aguileramp_oldschool_2e1e7540", "aguileramp_oldschool"),
    ("a_halo_so_bright_it_bleeds_3ee84993", "a_halo_so_bright"),
    ("blair_razor_project_e7dd3210", "blair_razor"),
    ("cyborg_5569c5cf", "cyborg"),
    ("dragonzv30amp_85acc35c", "dragonzv30amp"),
    ("engraved4_platinum_5638acf5", "engraved4_platinum"),
    ("goodgawd_bba84deb", "goodgawd"),
    ("infected_fx_gray_no_transparency_9f3bd211", "infected_fx_gray"),
    ("minimalistic_black_145917e6", "minimalistic_black"),
    ("rancid_amp_5_42a78437", "rancid_amp_5"),
    ("ruki2_by_michi_caa5bfe3", "ruki2_by_michi"),
    ("the_four_horsemen_523e6bdf", "the_four_horsemen"),
    ("tvxq_winamp_skins_by_roseweedy_c379f7bd", "tvxq_roseweedy"),
    ("zelda_amp_gold_3cc38af4", "zelda_amp_gold"),
]

# Easy->hard order (Gate-1 experience). Trim this list to run Gate 2 in waves.
EXPERTS = [
    "POSBAR", "PLAYPAUS", "MONOSTER", "SHUFREP", "VOLUME",
    "MAIN", "CBUTTONS", "PLEDIT", "BALANCE", "TITLEBAR", "EQMAIN",
]

# Shared model architecture (identical at L1 and adversarial stages).
BASE, ATTN_DIM, DEC_CH, HEADS, ATTN_LAYERS = 48, 256, 128, 4, 2
DECODER = "progressive"
SCALE = "gate2"            # ~459 variants/skin -> ~6.4k renders/BMP across 14 skins
BATCH = 4                 # FP32 base config fits T4 16GB at batch 4
PROGRESS_EVERY = 200
EVAL_MAX_ITEMS = 160      # periodic early-stop subset (sampled across skins)
EARLY_STOP_MAE, EARLY_STOP_HIT5, EARLY_STOP_PATIENCE = 0.008, 0.93, 2

# Per-stage caps (each <1h, the project rule). Multi-skin needs more steps than
# Gate-1, so L1 gets a longer cap; early-stop ends convergent experts sooner.
L1_STEPS, L1_LR, L1_MAX_MIN, L1_EVAL_EVERY = 40000, 3e-4, 45, 600
ADV_STEPS, ADV_LR, ADV_MAX_MIN, ADV_EVAL_EVERY = 30000, 1e-4, 35, 600
ADV_WEIGHT, FM_WEIGHT, D_LR = 0.02, 1.0, 2e-4

# Global wall-clock budget: stop launching new experts past this so output (and
# every checkpoint trained so far) persists before Kaggle's ~12h hard kill.
GLOBAL_BUDGET_MIN = 600

DATA_OUT = SCRATCH / "data_v10_gate2"
RUNS_ROOT = SCRATCH / "runs" / "v10_gate2"
CK_ROOT = SCRATCH / "v10_ck"     # staged best ckpt per expert (for the demo)
T_START = time.time()


def elapsed_min() -> float:
    return (time.time() - T_START) / 60.0


def run(label, cmd, cwd=None, capture=True):
    print(f"\n=== {label} (t+{elapsed_min():.1f}min) ===", flush=True)
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


def arch_args() -> list[str]:
    return [
        "--base", str(BASE), "--attn-dim", str(ATTN_DIM), "--dec-ch", str(DEC_CH),
        "--heads", str(HEADS), "--attn-layers", str(ATTN_LAYERS),
        "--query-div", "4", "--decoder", DECODER,
        "--eval-max-items", str(EVAL_MAX_ITEMS), "--progress-every", str(PROGRESS_EVERY),
        "--num-workers", "2", "--device", "cuda",
    ]


def eval_expert(stem: str, ckpt: Path, eval_dir: Path) -> dict:
    run(f"eval_{stem}", [
        sys.executable, "scripts/eval_bmp_expert.py",
        "--data", str(DATA_OUT), "--bmp", f"{stem}.bmp",
        "--checkpoint", str(ckpt), "--out", str(eval_dir),
        "--batch", "8", "--grid-samples", "16", "--device", "cuda",
    ], cwd=REPO)
    try:
        return json.loads((eval_dir / "metrics.json").read_text())
    except Exception as e:  # noqa: BLE001
        return {"error": str(e), "gate2_pass": False, "mae_mean": 9.9, "hit_5_255_mean": 0.0}


def train_expert(stem: str) -> dict:
    """L1 -> eval -> (adversarial if needed) -> eval. Stage the better ckpt."""
    bmp = f"{stem}.bmp"
    l1_out = RUNS_ROOT / f"{stem}_l1"
    l1_cmd = [
        sys.executable, "train_bmp_expert.py",
        "--data", str(DATA_OUT), "--bmp", bmp, "--out", str(l1_out),
        "--steps", str(L1_STEPS), "--batch", str(BATCH), "--lr", str(L1_LR),
        "--max-minutes", str(L1_MAX_MIN), "--eval-every", str(L1_EVAL_EVERY),
        "--checkpoint-every", "3000",
        "--early-stop", "--early-stop-mae", str(EARLY_STOP_MAE),
        "--early-stop-hit5", str(EARLY_STOP_HIT5), "--early-stop-patience", str(EARLY_STOP_PATIENCE),
        "--amp",
        *arch_args(),
    ]
    run(f"train_{stem}_L1", l1_cmd, cwd=REPO, capture=False)
    l1_best = l1_out / "best.safetensors"
    if not l1_best.exists():
        l1_best = l1_out / "last.safetensors"
    m = eval_expert(stem, l1_best, l1_out / "eval")
    best_ckpt, best_m, recipe = l1_best, m, "L1"

    # Escalate to adversarial only if L1 didn't clear every skin and there is
    # still time budget for a ~35min stage.
    if not m.get("gate2_pass") and elapsed_min() + ADV_MAX_MIN + 5 < GLOBAL_BUDGET_MIN:
        adv_out = RUNS_ROOT / f"{stem}_adv"
        adv_cmd = [
            sys.executable, "train_bmp_expert.py",
            "--data", str(DATA_OUT), "--bmp", bmp, "--out", str(adv_out),
            "--init-from", str(l1_best),
            "--steps", str(ADV_STEPS), "--batch", str(BATCH), "--lr", str(ADV_LR),
            "--max-minutes", str(ADV_MAX_MIN), "--eval-every", str(ADV_EVAL_EVERY),
            "--checkpoint-every", "2000",
            "--adversarial", "--adv-weight", str(ADV_WEIGHT),
            "--fm-weight", str(FM_WEIGHT), "--d-lr", str(D_LR),
            "--early-stop", "--early-stop-mae", str(EARLY_STOP_MAE),
            "--early-stop-hit5", str(EARLY_STOP_HIT5), "--early-stop-patience", str(EARLY_STOP_PATIENCE),
            *arch_args(),
        ]
        run(f"train_{stem}_ADV", adv_cmd, cwd=REPO, capture=False)
        adv_best = adv_out / "best.safetensors"
        if not adv_best.exists():
            adv_best = adv_out / "last.safetensors"
        am = eval_expert(stem, adv_best, adv_out / "eval")
        # Prefer the ckpt that passes gate2; else the lower mean mae.
        if (am.get("gate2_pass") and not best_m.get("gate2_pass")) or \
           (am.get("mae_mean", 9.9) < best_m.get("mae_mean", 9.9)):
            best_ckpt, best_m, recipe = adv_best, am, "L1+adversarial"

    # Stage the best ckpt for the demo + downstream gates.
    dst = CK_ROOT / stem
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_ckpt, dst / "last.safetensors")
    # Persist per-expert artifacts to the bundle.
    edst = OUT / "experts" / stem
    edst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_ckpt, edst / "best.safetensors")
    src_eval = (l1_out if recipe == "L1" else (RUNS_ROOT / f"{stem}_adv")) / "eval"
    for f in ("metrics.json", "pred_vs_target_grid.png", "per_variant.csv"):
        if (src_eval / f).exists():
            shutil.copy2(src_eval / f, edst / f)
    verdict = {
        "stem": stem, "recipe": recipe,
        "gate2_pass": bool(best_m.get("gate2_pass")),
        "mae_mean": round(best_m.get("mae_mean", 9.9), 5),
        "hit_5_255_mean": round(best_m.get("hit_5_255_mean", 0.0), 4),
        "n_skins": best_m.get("n_skins"), "worst_skin": best_m.get("worst_skin"),
        "per_skin_fail": [s for s, v in best_m.get("per_skin", {}).items() if not v.get("pass")],
    }
    print(f"\n>>> VERDICT {stem}: {'PASS' if verdict['gate2_pass'] else 'FAIL'} "
          f"recipe={recipe} mae={verdict['mae_mean']} hit5={verdict['hit_5_255_mean']} "
          f"worst={verdict['worst_skin']} fails={verdict['per_skin_fail']}\n", flush=True)
    return verdict


# ---- 1. environment ---------------------------------------------------------
summaries = []
summaries.append(run("01_clone", ["git", "clone", "--branch", REPO_BRANCH, REPO_URL, str(REPO)]))
summaries.append(run("02_rev", ["git", "-C", str(REPO), "rev-parse", "HEAD"]))
summaries.append(run("03_pip", [sys.executable, "-m", "pip", "install", "--quiet",
                                "safetensors", "pyyaml", "Pillow"]))

# ---- 2. generate the 14-skin dataset (append each skin into one dir) ---------
missing = [d for d, _ in SKINS if not (DATA_ROOT / d).exists()]
assert not missing, f"missing skin dirs on Kaggle: {missing}"
for i, (skin_dir, skin_id) in enumerate(SKINS):
    cmd = [
        sys.executable, "scripts/make_v10_bmp_expert_dataset.py",
        "--skin", str(DATA_ROOT / skin_dir), "--skin-id", skin_id,
        "--scale", SCALE, "--out", str(DATA_OUT), "--seed", str(i),
        "--progress-every", "200",
    ]
    if i > 0:
        cmd.append("--append")
    summaries.append(run(f"04_gen_{skin_id}", cmd, cwd=REPO))
# sanity: row counts per BMP after all skins appended
for stem in ("MAIN", "EQMAIN"):
    csv_p = DATA_OUT / "csv" / f"train_{stem}.csv"
    n = sum(1 for _ in csv_p.open()) - 1 if csv_p.exists() else 0
    print(f"sanity: train_{stem}.csv rows={n} (expect ~{len(SKINS)} skins worth)", flush=True)

# ---- 3. train each expert (L1 -> adversarial fallback), budget-guarded -------
verdicts = []
for stem in EXPERTS:
    if elapsed_min() + L1_MAX_MIN + 5 > GLOBAL_BUDGET_MIN:
        print(f"SKIP {stem}: global budget {GLOBAL_BUDGET_MIN}min nearly spent "
              f"(t+{elapsed_min():.1f}min)", flush=True)
        verdicts.append({"stem": stem, "recipe": "not_run", "gate2_pass": False})
        continue
    verdicts.append(train_expert(stem))
    (OUT / "verdicts.json").write_text(json.dumps(verdicts, indent=2))  # persist as we go

# ---- 4. demo compose with whatever experts trained --------------------------
demo_image = DATA_OUT / "renders" / "minimalistic_black_000000.png"
if demo_image.exists() and any(v.get("gate2_pass") is not None and v["recipe"] != "not_run" for v in verdicts):
    demo_out = SCRATCH / "v10_gate2_demo"
    summaries.append(run("05_demo", [
        sys.executable, "infer_v10.py", "--image", str(demo_image),
        "--checkpoints", str(CK_ROOT), "--out", str(demo_out),
        "--device", "cuda", "--demo-state",
    ], cwd=REPO))
    for f in ("side_by_side.png", "render_cranamp.png"):
        if (demo_out / f).exists():
            shutil.copy2(demo_out / f, OUT / f"demo_{f}")

# ---- 5. final verdict -------------------------------------------------------
n_pass = sum(1 for v in verdicts if v.get("gate2_pass"))
n_run = sum(1 for v in verdicts if v.get("recipe") != "not_run")
summary = {
    "name": "V10 Gate 2 (14 skins)", "skins": [s for _, s in SKINS],
    "experts_total": len(EXPERTS), "experts_run": n_run, "experts_pass": n_pass,
    "gate2_complete": bool(n_pass == len(EXPERTS)),
    "elapsed_min": round(elapsed_min(), 1),
    "verdicts": verdicts,
}
(OUT / "summary.json").write_text(json.dumps({"summary": summary, "runs": summaries}, indent=2))
print("\n=== V10 GATE 2 VERDICT ===", flush=True)
print(json.dumps(summary, indent=2), flush=True)
print("\nDONE.", flush=True)
