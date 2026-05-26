"""V7.1 Gate S2a: cross-skin StateFamilyExpander generalization (no oracle).

The deployability question: can the gated state-expander learn within-family
transitions that generalize to UNSEEN skin styles, using only the source frame
(no skin_id embedding)? S1 proved the mechanism on one skin; S2a tests transfer.

  skins:    all (~14) in the 16-skin set; 4 held out entirely (unseen style)
  task:     all mask_role=alternatives families, transitions only (--no-identity)
  head:     gated (copy + direct-write), gate_loss 0.05
  cond:     NO skin embedding (source frame carries the style)
  sampler:  family-balanced + difficulty weights + mixed local/global pairs
  splits:   train / seen_skin_unseen_pair / heldout_skin
  steps:    150k, snapshot+eval every 10k

Pass is judged on the HELD-OUT split's changed_hit5 (region-split eval). Oracle
skin embedding is reserved for an S2b diagnostic only if S2a fails.

NOTE: this kernel is built for review. It will run when pushed to Kaggle.
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
OUT = WORK / "s2_state_gated_outputs"
OUT.mkdir(parents=True, exist_ok=True)
DATA_ROOT = Path(
    "/kaggle/input/datasets/dmitriisamoilenko/uigen-data/uigen-data-v7-16skin/data_v7_16skin_completion"
)

# Held-out skins: chosen for genuinely non-degenerate transitions (real
# CBUTTONS pressed states + slider movement) so the unseen-style test is
# meaningful. The rest train.
HELDOUT_SKINS = [
    "the_four_horsemen_523e6bdf",
    "blair_razor_project_e7dd3210",
    "tvxq_winamp_skins_by_roseweedy_c379f7bd",
    "infected_fx_gray_no_transparency_9f3bd211",
]
SEEN_PAIR_VAL_FRACTION = 0.20
STEPS = 150000
BATCH = 64
SNAPSHOT_EVERY = 10000
GATE_LOSS_WEIGHT = 0.05
SPLIT_SEED = 0
FAMILY_WEIGHTS = "configs/state_expander_family_weights_s2.yaml"
RUN_OUT = WORK / "runs" / "v7_state_s2a_16skin_gated"


def run(label, cmd, cwd=None, capture=True):
    print(f"\n=== {label} ===", flush=True)
    print("$ " + " ".join(cmd), flush=True)
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

skin_dirs = sorted(p for p in DATA_ROOT.iterdir()
                   if p.is_dir() and not p.name.startswith("_"))
assert skin_dirs, f"no skin dirs under {DATA_ROOT}"
skin_sources = ",".join(f"{p.name}={p}" for p in skin_dirs)
all_ids = [p.name for p in skin_dirs]
missing = [s for s in HELDOUT_SKINS if s not in all_ids]
assert not missing, f"heldout skins absent from data: {missing}"
print(f"{len(all_ids)} skins; heldout {len(HELDOUT_SKINS)}: {HELDOUT_SKINS}", flush=True)

train_cmd = [
    sys.executable, "train_v7_state_expander.py",
    "--skin-sources", skin_sources,
    "--output-mode", "gated", "--no-identity",
    "--gate-loss-weight", str(GATE_LOSS_WEIGHT),
    "--heldout-skins", ",".join(HELDOUT_SKINS),
    "--seen-pair-val-fraction", str(SEEN_PAIR_VAL_FRACTION),
    "--family-weights", FAMILY_WEIGHTS,
    "--local-pair-prob", "0.5", "--local-delta", "2",
    "--base-channels", "48", "--steps", str(STEPS), "--batch", str(BATCH),
    "--lr", "1e-3", "--sobel-weight", "0.25",
    "--skin-embedding-dim", "0",  # S2a: NO oracle skin embedding
    "--checkpoint-every", str(SNAPSHOT_EVERY), "--snapshot-every", str(SNAPSHOT_EVERY),
    "--progress-every", "1000", "--seed", str(SPLIT_SEED),
    "--out", str(RUN_OUT), "--device", "cuda",
]
summaries.append(run("04_train", train_cmd, cwd=REPO, capture=False))

snaps = sorted(RUN_OUT.glob("snapshot_step*.safetensors"))
if (RUN_OUT / "last.safetensors").exists():
    snaps.append(RUN_OUT / "last.safetensors")
traj = []
for ck in snaps:
    outj = OUT / f"{ck.stem}.s2.json"
    summaries.append(run(f"05_eval_{ck.stem}", [
        sys.executable, "scripts/21_eval_v7_state_expander_s2.py",
        "--skin-sources", skin_sources, "--checkpoint", str(ck),
        "--heldout-skins", ",".join(HELDOUT_SKINS),
        "--seen-pair-val-fraction", str(SEEN_PAIR_VAL_FRACTION),
        "--split-seed", str(SPLIT_SEED), "--batch", "64", "--device", "cuda",
        "--out-json", str(outj),
    ], cwd=REPO))
    try:
        r = json.loads(outj.read_text())
        g = r["gate"]
        h = r["splits"].get("heldout", {}).get("aggregate", {})
        s = r["splits"].get("seen_val", {}).get("aggregate", {})
        traj.append({
            "snapshot": ck.stem,
            "verdict": "HARD" if g.get("hard_pass") else ("SOFT" if g.get("soft_pass") else "FAIL"),
            "heldout_mean_changed": h.get("mean_changed_hit5"),
            "heldout_min_family_changed": h.get("min_family_changed_hit5"),
            "heldout_unchanged_min": h.get("min_family_unchanged_hit5"),
            "heldout_gate_gap": h.get("mean_gate_gap"),
            "seenpair_mean_changed": s.get("mean_changed_hit5"),
            "families_below_0.85": g.get("families_below_0.85"),
        })
    except Exception as e:  # noqa: BLE001
        traj.append({"snapshot": ck.stem, "error": str(e)})

(OUT / "summary.json").write_text(json.dumps(
    {"heldout_skins": HELDOUT_SKINS, "steps": STEPS, "batch": BATCH,
     "runs": summaries, "eval_trajectory": traj}, indent=2))

print("\n=== S2a EVAL TRAJECTORY (held-out skin, region-split) ===", flush=True)
print(f"{'snapshot':22s} {'verdict':5s} {'held_chg_mean':>13s} {'held_chg_min':>12s} "
      f"{'held_unc_min':>12s} {'gap':>6s} {'seen_chg':>8s}", flush=True)
for e in traj:
    if "error" in e:
        print(f"{e['snapshot']:22s} ERROR {e['error']}", flush=True); continue
    g = lambda k: (e[k] if e[k] is not None else float("nan"))
    print(f"{e['snapshot']:22s} {e['verdict']:5s} {g('heldout_mean_changed'):13.3f} "
          f"{g('heldout_min_family_changed'):12.3f} {g('heldout_unchanged_min'):12.3f} "
          f"{g('heldout_gate_gap'):6.3f} {g('seenpair_mean_changed'):8.3f}", flush=True)
print("\nDONE.", flush=True)
