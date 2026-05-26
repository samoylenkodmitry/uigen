"""V7.1 Gate S2a-geomgate: cross-skin generalization via a geometry gate prior.

Both prior cross-skin attempts FAILED the same way: on a held-out (unseen-style)
skin the gate never opened (gate_gap ~0.1), so changed pixels were never written.
  - S2a (no context):   held changed_hit5 0.062 @150k, flat.
  - S2a-context (style): held changed_hit5 0.054 @50k,  flat (no lift-off).
Diagnosis: the gate is produced from RGB-entangled U-Net features, which are
out-of-distribution on a new style, so the model can't even LOCATE the change.

GPT-5.5 Pro's fix (this run): a SKIN-INDEPENDENT geometry gate prior. The gate
logits become content_gate_logits + geometry_gate_logits, where the geometry
branch is a per-pixel coordinate MLP over fixed classic sprite geometry
(source/target rects, frame indices, file size) + family id — NO RGB, NO skin
id. In classic skins this geometry is identical across skins, so "where the
change is" can be learned in a style-invariant way. The RGB head still learns
WHAT to write. Tests whether the dead held-out gate localization is the basic
blocker.

This kernel is byte-identical to the failed S2a runs except:
  - --geometry-gate --geo-gate-hidden 64   (the one architecture change)
  - --style-context-dim 0, --skin-embedding-dim 0  (clean: geometry only)
  - STEPS = 50000   (early read; prior held-out lines were flat throughout)

Decision rule (read the trajectory table):
  * LIFT-OFF: held-out changed_hit5 rises well above the prior flat ~0.05-0.06
    AND gate_gap opens (the geometry prior is opening the gate on unseen skins)
    -> extend to 150k.
  * FLAT: held-out still ~0.05-0.10, gate_gap ~0 -> geometry alone can't
    localize the change; revisit (the gate opens but RGB wrong is a DIFFERENT,
    later problem — here we are still failing to open it).

Same 14 skins, same 4 held out, same sampler / gate loss 0.05 / eval snapshots.
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
OUT = WORK / "s2_state_geomgate_outputs"
OUT.mkdir(parents=True, exist_ok=True)
DATA_ROOT = Path(
    "/kaggle/input/datasets/dmitriisamoilenko/uigen-data/uigen-data-v7-16skin/data_v7_16skin_completion"
)

# Identical held-out set to the failed S2a runs, for a clean comparison.
HELDOUT_SKINS = [
    "the_four_horsemen_523e6bdf",
    "blair_razor_project_e7dd3210",
    "tvxq_winamp_skins_by_roseweedy_c379f7bd",
    "infected_fx_gray_no_transparency_9f3bd211",
]
SEEN_PAIR_VAL_FRACTION = 0.20
STEPS = 50000          # early read (prior held-out lines were flat the whole run)
BATCH = 64
SNAPSHOT_EVERY = 10000
GATE_LOSS_WEIGHT = 0.05
GEO_GATE_HIDDEN = 64    # THE one architecture change vs failed S2a
SPLIT_SEED = 0
FAMILY_WEIGHTS = "configs/state_expander_family_weights_s2.yaml"
RUN_OUT = WORK / "runs" / "v7_state_s2a_geomgate_16skin"


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
print(f"{len(all_ids)} skins; heldout {len(HELDOUT_SKINS)}: {HELDOUT_SKINS}; "
      f"geo_gate_hidden={GEO_GATE_HIDDEN}", flush=True)

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
    "--skin-embedding-dim", "0",            # NO oracle skin embedding
    "--style-context-dim", "0",             # NO style code (geometry only)
    "--geometry-gate", "--geo-gate-hidden", str(GEO_GATE_HIDDEN),  # the change
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
     "geo_gate_hidden": GEO_GATE_HIDDEN,
     "runs": summaries, "eval_trajectory": traj}, indent=2))

print("\n=== S2a-GEOMGATE EVAL TRAJECTORY (held-out skin, region-split) ===", flush=True)
print("compare held_chg_mean against prior flat lines: S2a 0.062@150k, context 0.054@50k", flush=True)
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
