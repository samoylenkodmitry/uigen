"""V7.1 Gate S1: StateFamilyExpander convergence run on T4 (one skin, gated).

Validates that the gated state-expander can learn true alternatives-family
transitions to the S1 bar on a single, non-degenerate skin (dragonzv30amp,
CBUTTONS 6/6 real + all sliders/toggles real). This is a state-expander
CAPACITY test, NOT a full V7 solution.

  skin:    dragonzv30amp (one skin)
  task:    all mask_role=alternatives families, transitions only (--no-identity)
  head:    gated (copy-biased gate + direct write)
  steps:   100k, family-balanced sampling, snapshot+eval every 10k
  eval:    region-split (support / changed / unchanged) + gate diagnostics,
           per family. S1 verdict is region-aware (changed_hit5 is what proves
           the model learned transitions vs just copying).

GATE_LOSS_WEIGHT is set from the local 10k A/B (gated no-gate-loss vs
gate_loss=0.05), chosen by changed_hit5.
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
OUT = WORK / "s1_state_gated_outputs"
OUT.mkdir(parents=True, exist_ok=True)

DATA_ROOT = Path(
    "/kaggle/input/datasets/dmitriisamoilenko/uigen-data/uigen-data-v7-16skin/data_v7_16skin_completion"
)
SKIN = "dragonzv30amp_85acc35c"

# Run config (the S1 convergence recipe).
STEPS = 100000
BATCH = 64
SNAPSHOT_EVERY = 10000
GATE_LOSS_WEIGHT = 0.05  # chosen by local 10k A/B (changed_hit5 0.627 vs 0.386 at 0.0)
RUN_OUT = WORK / "runs" / "v7_state_s1_gated_dragon_100k"


def run(label: str, cmd: list[str], cwd: Path | None = None, env: dict | None = None,
        capture: bool = True) -> dict:
    print(f"\n=== {label} ===", flush=True)
    print("$ " + " ".join(cmd), flush=True)
    t0 = time.time()
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    if capture:
        res = subprocess.run(cmd, cwd=cwd, env=full_env, capture_output=True, text=True)
        (OUT / f"{label}.stdout.txt").write_text(res.stdout)
        (OUT / f"{label}.stderr.txt").write_text(res.stderr)
        print(res.stdout[-3000:], flush=True)
        if res.returncode != 0:
            print(res.stderr[-3000:], flush=True)
        return {"label": label, "rc": res.returncode, "dur_sec": round(time.time() - t0, 1)}
    res = subprocess.run(cmd, cwd=cwd, env=full_env)
    return {"label": label, "rc": res.returncode, "dur_sec": round(time.time() - t0, 1)}


summaries: list[dict] = []
summaries.append(run("01_clone", ["git", "clone", "--branch", REPO_BRANCH, REPO_URL, str(REPO)]))
summaries.append(run("02_rev", ["git", "-C", str(REPO), "rev-parse", "HEAD"]))
summaries.append(run("03_pip", [sys.executable, "-m", "pip", "install", "--quiet",
                                "safetensors", "pyyaml", "Pillow"]))

skin_dir = DATA_ROOT / SKIN
assert skin_dir.is_dir(), f"missing skin dir {skin_dir}"
print(f"skin source: {skin_dir}", flush=True)

train_cmd = [
    sys.executable, "train_v7_state_expander.py",
    "--skin-sources", f"{SKIN}={skin_dir}",
    "--output-mode", "gated",
    "--no-identity",
    "--base-channels", "48",
    "--steps", str(STEPS),
    "--batch", str(BATCH),
    "--lr", "1e-3",
    "--sobel-weight", "0.25",
    "--gate-loss-weight", str(GATE_LOSS_WEIGHT),
    "--checkpoint-every", str(SNAPSHOT_EVERY),
    "--snapshot-every", str(SNAPSHOT_EVERY),
    "--progress-every", "1000",
    "--out", str(RUN_OUT),
    "--device", "cuda", "--seed", "0",
]
# capture=False so the live progress + per-family + gate lines stream to stdout.
summaries.append(run("04_train", train_cmd, cwd=REPO, capture=False))

# Per-snapshot region-split eval (the metric that distinguishes learning from copying).
snaps = sorted(RUN_OUT.glob("snapshot_step*.safetensors"))
if (RUN_OUT / "last.safetensors").exists():
    snaps.append(RUN_OUT / "last.safetensors")
eval_index: list[dict] = []
for ck in snaps:
    label = f"05_eval_{ck.stem}"
    outj = OUT / f"{ck.stem}.eval.json"
    s = run(label, [
        sys.executable, "scripts/20_eval_v7_state_expander.py",
        "--skin-sources", f"{SKIN}={skin_dir}",
        "--checkpoint", str(ck),
        "--batch", "64", "--device", "cuda",
        "--out-json", str(outj),
    ], cwd=REPO)
    summaries.append(s)
    try:
        r = json.loads(outj.read_text())
        ag = r["aggregate"]; g = r["gate"]
        eval_index.append({
            "snapshot": ck.stem,
            "verdict": "HARD" if g["hard_pass"] else ("SOFT" if g["soft_pass"] else "FAIL"),
            "mean_support_hit5": ag["mean_support_hit5"],
            "min_support_hit5": ag["min_family_support_hit5"],
            "mean_changed_hit5": ag["mean_changed_hit5"],
            "min_changed_hit5": ag["min_family_changed_hit5"],
            "min_unchanged_hit5": ag["min_family_unchanged_hit5"],
            "mean_gate_gap": ag["mean_gate_gap"],
        })
    except Exception as e:  # noqa: BLE001
        eval_index.append({"snapshot": ck.stem, "error": str(e)})

(OUT / "summary.json").write_text(json.dumps(
    {"config": {"skin": SKIN, "steps": STEPS, "batch": BATCH,
                "gate_loss_weight": GATE_LOSS_WEIGHT},
     "runs": summaries, "eval_trajectory": eval_index}, indent=2))

print("\n=== S1 EVAL TRAJECTORY (region-split) ===", flush=True)
print(f"{'snapshot':22s} {'verdict':6s} {'sup_mean':>8s} {'sup_min':>8s} "
      f"{'chg_mean':>8s} {'chg_min':>8s} {'unc_min':>8s} {'gate_gap':>8s}", flush=True)
for e in eval_index:
    if "error" in e:
        print(f"{e['snapshot']:22s} ERROR {e['error']}", flush=True)
        continue
    print(f"{e['snapshot']:22s} {e['verdict']:6s} {e['mean_support_hit5']:8.3f} "
          f"{e['min_support_hit5']:8.3f} {e['mean_changed_hit5']:8.3f} {e['min_changed_hit5']:8.3f} "
          f"{e['min_unchanged_hit5']:8.3f} {e['mean_gate_gap']:8.3f}", flush=True)
print("\nDONE.", flush=True)
