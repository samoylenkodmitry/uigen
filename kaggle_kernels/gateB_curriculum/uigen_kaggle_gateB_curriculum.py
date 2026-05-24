"""V7 Gate B staged curriculum (Phases A -> B -> C) on T4.

Phase A (30k): c48 from scratch. state_family-only mask. BV-heavy
    weights (BALANCE/VOLUME 8 vs other files 1-2). Replace mode so
    weight-0 files would be dropped — but every file is kept at >= 1
    so nothing regresses while BV dominates the gradient budget.

Phase B (30k): resumes Phase A. Same state_family-only mask. Broader
    weights (BV down to 4, chrome/strip files up to 2-3). Consolidates
    every file before mixed-mode introduction.

Phase C (40k): resumes Phase B. Mixed mask: state_family 0.60,
    random_rect 0.30, provenance 0.10. Same weights as Phase B. lr
    halved so the mode mix doesn't blow away Phase B's state_family
    fit. This is the Gate B test — mae<0.015, hit5>0.90 on mixed eval.

Total: 100k optimizer steps. At ~0.15 s/step on T4 this is ~250 min
train + per-phase eval + final per-snapshot eval + diff dump.
Well within the 9-hour T4 session limit.

Eval cadence: per snapshot (every 10k steps) for state_family and mix.
After Phase C, dump target/pred/diff panels for the worst 3 skins on
BALANCE+VOLUME+POSBAR+MONOSTER+PLAYPAUS so we can see remaining
failure modes.

Trainer telemetry: prints by_mode / by_file / by_skin loss breakdowns
each progress window, and an "effective file probabilities" line per
phase showing the post-normalization sampling distribution. Replace
mode means absent files have probability 0 — verifiable in that log.
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
OUT = WORK / "gateB_curriculum_outputs"
OUT.mkdir(parents=True, exist_ok=True)

DATA_ROOT = Path(
    "/kaggle/input/datasets/dmitriisamoilenko/uigen-data/uigen-data-v7-16skin/data_v7_16skin_completion"
)

PHASES = [
    ("phaseA", "experiments/v7_completer_gateB_phaseA.yaml"),
    ("phaseB", "experiments/v7_completer_gateB_phaseB.yaml"),
    ("phaseC", "experiments/v7_completer_gateB_phaseC.yaml"),
]

NUM_WORST_SKINS = 3
DIAGNOSTIC_FILES = ("BALANCE.bmp", "VOLUME.bmp", "POSBAR.bmp",
                    "MONOSTER.bmp", "PLAYPAUS.bmp")


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


def _train_phase(phase_label: str, yaml_path: str) -> dict:
    """Run one phase via run_experiment.py. capture=False so the
    trainer's progress + per-mode/per-file/per-skin breakdowns + the
    effective-file-probabilities log line show up live."""
    return run(f"06_train_{phase_label}", [
        sys.executable, "scripts/run_experiment.py",
        "--experiment", yaml_path,
    ], cwd=REPO, env=env_for_make, capture=False)


# Dry-runs first to catch any path-resolution issue early.
for phase_label, yaml_path in PHASES:
    summaries.append(run(f"05_dry_{phase_label}", [
        sys.executable, "scripts/run_experiment.py",
        "--experiment", yaml_path,
        "--dry-run",
    ], cwd=REPO, env=env_for_make))

# ----- GUARDRAIL: pre-launch verification --------------------------------
# Print and validate the effective per-phase recipe BEFORE any training
# starts. Fail loud if the file probabilities or mask weights don't match
# the planned curriculum — easier to spot a yaml typo here than at hour 4.
print("\n=== GUARDRAIL: pre-launch verification ===", flush=True)
import yaml as _yaml
sys.path.insert(0, str(REPO))
from train_v7_completer import build_file_weights

EXPECTED_RUN_DIRS = {
    "phaseA": "gateB_curriculum_phaseA",
    "phaseB": "gateB_curriculum_phaseB",
    "phaseC": "gateB_curriculum_phaseC",
}
EXPECTED_MASK = {
    "phaseA": {"state_family": 1.0, "random_rect": 0.0,
               "provenance": 0.0, "whole_file": 0.0, "passthrough": 0.0},
    "phaseB": {"state_family": 1.0, "random_rect": 0.0,
               "provenance": 0.0, "whole_file": 0.0, "passthrough": 0.0},
    "phaseC": {"state_family": 0.60, "random_rect": 0.30,
               "provenance": 0.10, "whole_file": 0.0, "passthrough": 0.0},
}

ALL_FILES = ("MAIN.bmp", "TITLEBAR.bmp", "PLEDIT.bmp", "EQMAIN.bmp",
             "CBUTTONS.bmp", "SHUFREP.bmp", "POSBAR.bmp", "PLAYPAUS.bmp",
             "MONOSTER.bmp", "BALANCE.bmp", "VOLUME.bmp")


def _normalize(weights: dict) -> dict[str, float]:
    total = sum(w for w in weights.values() if w > 0)
    if total <= 0:
        return {k: 0.0 for k in weights}
    return {k: (w / total if w > 0 else 0.0) for k, w in weights.items()}


guardrail_violations: list[str] = []

for phase_label, yaml_path in PHASES:
    spec = _yaml.safe_load((REPO / yaml_path).read_text(encoding="utf-8"))
    args = spec.get("args") or {}
    out = spec.get("out", "")
    print(f"\n-- {phase_label} ({yaml_path}) --", flush=True)
    print(f"  out: {out}", flush=True)
    print(f"  steps: {args.get('steps')}  batch: {args.get('batch')}  "
          f"lr: {args.get('lr')}", flush=True)
    print(f"  resume-from: {args.get('resume-from', '(scratch)')}", flush=True)

    # Run dir check.
    exp_dir = EXPECTED_RUN_DIRS[phase_label]
    if exp_dir not in out:
        guardrail_violations.append(
            f"{phase_label} out={out!r} does not contain expected dir {exp_dir!r}"
        )

    # Mask weights check.
    mask_actual = {
        "state_family": float(args.get("mask-state-family", 0)),
        "random_rect":  float(args.get("mask-random-rect", 0)),
        "provenance":   float(args.get("mask-provenance", 0)),
        "whole_file":   float(args.get("mask-whole-file", 0)),
        "passthrough":  float(args.get("mask-passthrough", 0)),
    }
    print(f"  mask weights: {mask_actual}", flush=True)
    exp_mask = EXPECTED_MASK[phase_label]
    for k, v in exp_mask.items():
        if abs(mask_actual[k] - v) > 1e-6:
            guardrail_violations.append(
                f"{phase_label} mask {k}: expected {v}, got {mask_actual[k]}"
            )

    # File weights check (post-normalization, replace mode).
    weights_yaml_rel = args.get("file-sampling-weights")
    weights_mode = args.get("file-sampling-weights-mode", "merge")
    print(f"  file-sampling-weights: {weights_yaml_rel}  mode: {weights_mode}", flush=True)
    if weights_mode != "replace":
        guardrail_violations.append(
            f"{phase_label} file-sampling-weights-mode={weights_mode}, expected 'replace'"
        )
    raw = build_file_weights(REPO / weights_yaml_rel, mode=weights_mode)
    norm = _normalize(raw)
    print(f"  effective file probabilities (post-normalization):", flush=True)
    for fn in sorted(raw, key=lambda k: -raw[k]):
        print(f"    {fn:20s} raw={raw[fn]:.3f}  prob={norm[fn]:.4f}", flush=True)

    # Every file with weight 0 in replace mode would be dropped. The whole
    # *point* of this curriculum is that no file regresses, so every file
    # must be present with weight > 0 in every phase.
    missing = [fn for fn in ALL_FILES if raw.get(fn, 0) <= 0]
    if missing:
        guardrail_violations.append(
            f"{phase_label} missing nonzero weight for: {missing}"
        )

    # Phase A specifically must have BV strictly larger than every other
    # file's weight (the BV-heavy invariant).
    if phase_label == "phaseA":
        bv_min = min(raw.get("BALANCE.bmp", 0), raw.get("VOLUME.bmp", 0))
        for fn in ALL_FILES:
            if fn in ("BALANCE.bmp", "VOLUME.bmp"):
                continue
            if raw.get(fn, 0) >= bv_min:
                guardrail_violations.append(
                    f"phaseA invariant: {fn} weight {raw[fn]} >= BV weight {bv_min}"
                )

# Distinct run dirs check.
out_dirs = {p: _yaml.safe_load((REPO / y).read_text(encoding="utf-8")).get("out", "")
            for p, y in PHASES}
if len(set(out_dirs.values())) != len(out_dirs):
    guardrail_violations.append(f"non-distinct run dirs: {out_dirs}")

print("\n-- guardrail summary --", flush=True)
if guardrail_violations:
    print("GUARDRAIL FAILED. Refusing to launch training:", flush=True)
    for v in guardrail_violations:
        print(f"  ! {v}", flush=True)
    raise SystemExit("guardrail violations — fix the yamls and re-push")
print("GUARDRAIL OK. All three phases pass pre-launch checks.", flush=True)
print(f"  Phase A: BV-heavy, every file nonzero, state_family=1.0", flush=True)
print(f"  Phase B: broader, every file nonzero, state_family=1.0", flush=True)
print(f"  Phase C: same weights as B, mixed mask "
      f"(sf=0.6/rr=0.3/pv=0.1)", flush=True)
# ----- end GUARDRAIL ------------------------------------------------------

# Train the three phases sequentially. Phase B and C have resume-from
# baked into their YAMLs pointing at the previous phase's @runs/.../
# last.safetensors path, which the runtime resolves to
# /kaggle/working/runs/gateB_curriculum_phaseX/last.safetensors.
for phase_label, yaml_path in PHASES:
    summaries.append(_train_phase(phase_label, yaml_path))

skin_args = ",".join(
    f"{d.name}={d}" for d in sorted(DATA_ROOT.iterdir()) if d.is_dir()
)
print(f"\nskin_sources count: {skin_args.count('=')}", flush=True)


def _eval(phase: str, snap: Path, mode: str, out_json: Path) -> dict:
    if mode == "state_family":
        mask_flags = ["--mask-provenance", "0", "--mask-state-family", "1",
                      "--mask-random-rect", "0", "--mask-whole-file", "0",
                      "--mask-passthrough", "0"]
    else:  # "mix" — default trainer mix
        mask_flags = []
    label = f"07_eval_{phase}_{mode}_{snap.stem}"
    return run(label, [
        sys.executable, "scripts/19_eval_v7_completer.py",
        "--state-families", "configs/state_families_classic.yaml",
        "--skin-sources", skin_args,
        "--checkpoint", str(snap),
        "--batch", "4", "--mask-samples", "4", "--device", "cuda",
        "--out-json", str(out_json),
        *mask_flags,
    ], cwd=REPO, env=env_for_make)


eval_index: list[dict] = []
for phase_label, _ in PHASES:
    run_dir = RUNS_DIR / f"gateB_curriculum_{phase_label}"
    print(f"\n=== {phase_label} run dir contents ===", flush=True)
    for p in sorted(run_dir.glob("*"))[:50]:
        print(f"  {p.name}  ({p.stat().st_size} B)", flush=True)
    snapshots = sorted(run_dir.glob("snapshot_step*.safetensors"))
    last_ckpt = run_dir / "last.safetensors"
    if last_ckpt.exists():
        snapshots.append(last_ckpt)
    for snap in snapshots:
        for mode in ("state_family", "mix"):
            out_json = OUT / f"eval_{phase_label}_{mode}_{snap.stem}.json"
            s = _eval(phase_label, snap, mode, out_json)
            summaries.append(s)
            if out_json.exists():
                data = json.loads(out_json.read_text())
                eval_index.append({
                    "phase": phase_label,
                    "snapshot": snap.name,
                    "mode": mode,
                    "aggregate": data.get("aggregate"),
                    "per_file": data.get("per_file"),
                    "per_skin": data.get("per_skin"),
                })


def _worst_skins(per_skin: dict, n: int) -> list[str]:
    if not isinstance(per_skin, dict):
        return []
    rows = sorted(
        ((sid, v.get("supported_mae")) for sid, v in per_skin.items()
         if isinstance(v, dict) and v.get("supported_mae") is not None),
        key=lambda kv: -float(kv[1]),
    )
    return [s for s, _ in rows[:n]]


# Pick the final Phase C mix-eval worst 3 skins for diff dump (mixed mode
# is the gate metric, so its per-skin worst is the most informative).
last_phaseC_mix = next(
    (r for r in reversed(eval_index)
     if r["phase"] == "phaseC" and r["mode"] == "mix" and r["snapshot"] == "last.safetensors"),
    None,
)
worst = _worst_skins(last_phaseC_mix.get("per_skin", {}) if last_phaseC_mix else {}, NUM_WORST_SKINS)
print(f"\n=== worst {NUM_WORST_SKINS} skins (Phase C, mix, final) ===", flush=True)
print(json.dumps(worst, indent=2), flush=True)

phaseC_last = RUNS_DIR / "gateB_curriculum_phaseC" / "last.safetensors"
diff_dir = OUT / "diffs"
diff_dir.mkdir(exist_ok=True)
if worst and phaseC_last.exists():
    summaries.append(run("08_dump_diffs_state_family", [
        sys.executable, "scripts/dump_v7_completer_diffs.py",
        "--state-families", "configs/state_families_classic.yaml",
        "--skin-sources", skin_args,
        "--checkpoint", str(phaseC_last),
        "--files", ",".join(DIAGNOSTIC_FILES),
        "--skins", ",".join(worst),
        "--mask-mode", "state_family",
        "--num-seeds", "2",
        "--device", "cuda",
        "--out-dir", str(diff_dir / "state_family"),
    ], cwd=REPO, env=env_for_make))
    summaries.append(run("09_dump_diffs_random_rect", [
        sys.executable, "scripts/dump_v7_completer_diffs.py",
        "--state-families", "configs/state_families_classic.yaml",
        "--skin-sources", skin_args,
        "--checkpoint", str(phaseC_last),
        "--files", ",".join(DIAGNOSTIC_FILES),
        "--skins", ",".join(worst),
        "--mask-mode", "random_rect",
        "--num-seeds", "2",
        "--device", "cuda",
        "--out-dir", str(diff_dir / "random_rect"),
    ], cwd=REPO, env=env_for_make))

import shutil
keep = WORK / "kept_artefacts"
keep.mkdir(exist_ok=True)
for phase_label, _ in PHASES:
    run_dir = RUNS_DIR / f"gateB_curriculum_{phase_label}"
    p_keep = keep / phase_label
    p_keep.mkdir(exist_ok=True)
    for name in ("metrics.jsonl", "config.json", "manifest.json",
                 "last.safetensors", "best.safetensors"):
        src = run_dir / name
        if src.exists():
            shutil.copy2(src, p_keep / name)
    for snap in run_dir.glob("snapshot_step*.safetensors"):
        shutil.copy2(snap, p_keep / snap.name)

(OUT / "summary.json").write_text(json.dumps({
    "steps": summaries,
    "eval_index": eval_index,
    "worst_skins_phaseC_mix": worst,
}, indent=2))

print("\n=== per-phase per-mode aggregate trajectory ===", flush=True)
print(f"{'phase':10s} {'snapshot':35s} {'mode':14s} {'mae':>9s} {'hit5':>8s}", flush=True)
for row in eval_index:
    agg = row.get("aggregate") or {}
    mae = agg.get("supported_mae", float('nan'))
    hit5 = agg.get("hit5", float('nan'))
    print(f"  {row['phase']:8s} {row['snapshot']:33s} {row['mode']:14s} {mae:9.5f} {hit5:8.4f}", flush=True)

# Headline per_file tables for the FINAL Phase C eval (state_family + mix).
for mode in ("state_family", "mix"):
    final = next(
        (r for r in reversed(eval_index)
         if r["phase"] == "phaseC" and r["mode"] == mode and r["snapshot"] == "last.safetensors"),
        None,
    )
    if final is None: continue
    pf = final.get("per_file") or {}
    items = list(pf.items()) if isinstance(pf, dict) else [(r.get("file", "?"), r) for r in pf]
    items.sort(key=lambda kv: -(kv[1].get("supported_mae", 0) if isinstance(kv[1], dict) else 0))
    print(f"\n=== final Phase C {mode} per_file ===", flush=True)
    for k, v in items:
        if not isinstance(v, dict): continue
        print(f"  {k:30s} mae={v.get('supported_mae'):.5f}  hit5={v.get('hit5'):.4f}", flush=True)
    ps = final.get("per_skin") or {}
    items = list(ps.items()) if isinstance(ps, dict) else [(r.get("skin", "?"), r) for r in ps]
    items.sort(key=lambda kv: -(kv[1].get("supported_mae", 0) if isinstance(kv[1], dict) else 0))
    print(f"\n=== final Phase C {mode} per_skin (top 6 worst) ===", flush=True)
    for k, v in items[:6]:
        if not isinstance(v, dict): continue
        print(f"  {k:42s} mae={v.get('supported_mae'):.5f}  hit5={v.get('hit5'):.4f}", flush=True)

# Gate check.
gate_pass = False
gate_reasons: list[str] = []
last_C_mix = last_phaseC_mix
last_C_sf = next(
    (r for r in reversed(eval_index)
     if r["phase"] == "phaseC" and r["mode"] == "state_family" and r["snapshot"] == "last.safetensors"),
    None,
)
if last_C_mix and last_C_sf:
    mix_agg = last_C_mix.get("aggregate") or {}
    sf_agg = last_C_sf.get("aggregate") or {}
    mix_mae = mix_agg.get("supported_mae", 1.0)
    mix_h5 = mix_agg.get("hit5", 0.0)
    sf_pf = last_C_sf.get("per_file") or {}
    bal = sf_pf.get("BALANCE.bmp", {}) if isinstance(sf_pf, dict) else {}
    vol = sf_pf.get("VOLUME.bmp", {}) if isinstance(sf_pf, dict) else {}
    pos = sf_pf.get("POSBAR.bmp", {}) if isinstance(sf_pf, dict) else {}
    mon = sf_pf.get("MONOSTER.bmp", {}) if isinstance(sf_pf, dict) else {}
    pp  = sf_pf.get("PLAYPAUS.bmp", {}) if isinstance(sf_pf, dict) else {}
    if mix_mae >= 0.015: gate_reasons.append(f"mix mae {mix_mae:.4f} >= 0.015")
    if mix_h5 <= 0.90:   gate_reasons.append(f"mix hit5 {mix_h5:.4f} <= 0.90")
    if bal.get("supported_mae", 1) >= 0.010 or bal.get("hit5", 0) <= 0.90:
        gate_reasons.append(f"BALANCE sf mae={bal.get('supported_mae')} hit5={bal.get('hit5')}")
    if vol.get("supported_mae", 1) >= 0.010 or vol.get("hit5", 0) <= 0.90:
        gate_reasons.append(f"VOLUME sf mae={vol.get('supported_mae')} hit5={vol.get('hit5')}")
    for fn, d in (("POSBAR", pos), ("MONOSTER", mon), ("PLAYPAUS", pp)):
        if d.get("supported_mae", 1) >= 0.05:
            gate_reasons.append(f"{fn} sf mae={d.get('supported_mae')} (>0.05)")
    gate_pass = not gate_reasons

print("\n=== GATE B ===", flush=True)
print("PASS" if gate_pass else "FAIL", flush=True)
for r in gate_reasons:
    print(f"  - {r}", flush=True)

all_ok = all(s["rc"] == 0 for s in summaries)
print("\nGATEB_CURRICULUM_RESULT:", "TRAINED" if all_ok else "FAIL", flush=True)
if not all_ok:
    raise SystemExit(1)
