#!/usr/bin/env python3
"""Portable V10 gate sweep: train N BMP experts on a (multi-skin) dataset with
the validated triage — L1 first, adversarial fine-tune only if a skin fails.

Machine-agnostic (local box, Lightning Studio, Kaggle): point it at a dataset
built by make_v10_bmp_expert_dataset.py and it trains every requested expert,
evals per-skin, stages the best checkpoint, and writes a summary. Batch size
auto-scales from GPU VRAM. Honors the project's progress-to-stdout rule via the
underlying trainer (--progress-every) and its own per-stage banners.

Example (Lightning / local):
    python scripts/v10_gate_sweep.py --data data_v10_gate2 --out runs/gate2 \
        --device cuda --experts ALL
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

ALL_EXPERTS = [  # easy -> hard (Gate-1 experience)
    "POSBAR", "PLAYPAUS", "MONOSTER", "SHUFREP", "VOLUME",
    "MAIN", "CBUTTONS", "PLEDIT", "BALANCE", "TITLEBAR", "EQMAIN",
]

# Default shared architecture (identical at L1 and adversarial stages). Override
# via --base/--attn-dim/--dec-ch/--attn-layers/--heads when a multi-skin run
# plateaus and needs more capacity (Gate 2 stresses the model far more than the
# one-skin Gate 1, so capacity is the first escalation lever).
DEF_ARCH = dict(base=48, attn_dim=256, dec_ch=128, heads=4, attn_layers=2)


def auto_batch(device: str) -> int:
    """Pick a safe batch from total VRAM. FP32-ish base config: ~3GB/sample
    headroom budget. AMP roughly halves it but we stay conservative."""
    if device != "cuda":
        return 2
    try:
        import torch
        gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:  # noqa: BLE001
        return 4
    if gib <= 10:
        return 2
    if gib <= 18:
        return 4
    if gib <= 26:
        return 8
    return 12


def run(label: str, cmd: list[str], log_dir: Path, capture: bool = False) -> int:
    print(f"\n=== {label} ===\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    t0 = time.time()
    if capture:
        res = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        (log_dir / f"{label}.out.txt").write_text(res.stdout)
        (log_dir / f"{label}.err.txt").write_text(res.stderr)
        print(res.stdout[-2500:], flush=True)
        if res.returncode != 0:
            print(res.stderr[-2500:], flush=True)
    else:
        res = subprocess.run(cmd, cwd=REPO)
    print(f"[{label}] rc={res.returncode} {time.time() - t0:.0f}s", flush=True)
    return res.returncode


def eval_expert(stem: str, data: Path, ckpt: Path, out: Path, device: str) -> dict:
    run(f"eval_{stem}", [
        sys.executable, "scripts/eval_bmp_expert.py",
        "--data", str(data), "--bmp", f"{stem}.bmp",
        "--checkpoint", str(ckpt), "--out", str(out),
        "--batch", "8", "--grid-samples", "16", "--device", device,
    ], out.parent, capture=True)
    try:
        return json.loads((out / "metrics.json").read_text())
    except Exception as e:  # noqa: BLE001
        return {"error": str(e), "gate2_pass": False, "gate1_pass": False, "mae_mean": 9.9}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Dataset dir (make_v10_bmp_expert_dataset.py output).")
    ap.add_argument("--out", required=True, help="Run root (per-expert dirs + ckpts + summary).")
    ap.add_argument("--experts", default="ALL", help="'ALL' or comma list, e.g. MAIN,EQMAIN.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=0, help="0 = auto from VRAM.")
    ap.add_argument("--l1-max-min", type=float, default=45.0)
    ap.add_argument("--adv-max-min", type=float, default=35.0)
    ap.add_argument("--global-budget-min", type=float, default=0.0,
                    help="0 = no global cap; else stop launching experts past it.")
    ap.add_argument("--l1-steps", type=int, default=40000)
    ap.add_argument("--adv-steps", type=int, default=30000)
    ap.add_argument("--eval-every", type=int, default=600)
    ap.add_argument("--eval-max-items", type=int, default=160)
    ap.add_argument("--no-amp", action="store_true", help="Disable AMP for the L1 stage.")
    ap.add_argument("--base", type=int, default=DEF_ARCH["base"])
    ap.add_argument("--attn-dim", type=int, default=DEF_ARCH["attn_dim"])
    ap.add_argument("--dec-ch", type=int, default=DEF_ARCH["dec_ch"])
    ap.add_argument("--heads", type=int, default=DEF_ARCH["heads"])
    ap.add_argument("--attn-layers", type=int, default=DEF_ARCH["attn_layers"])
    args = ap.parse_args()
    arch = ["--base", str(args.base), "--attn-dim", str(args.attn_dim),
            "--dec-ch", str(args.dec_ch), "--heads", str(args.heads),
            "--attn-layers", str(args.attn_layers), "--query-div", "4",
            "--decoder", "progressive"]

    data = Path(args.data)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ck_root = out / "ckpts"; ck_root.mkdir(exist_ok=True)
    experts = ALL_EXPERTS if args.experts.upper() == "ALL" else \
        [e.strip().upper() for e in args.experts.split(",") if e.strip()]
    batch = args.batch or auto_batch(args.device)
    t_start = time.time()

    def elapsed_min() -> float:
        return (time.time() - t_start) / 60.0

    print(f"V10 gate sweep: data={data} experts={experts} device={args.device} "
          f"batch={batch} amp={not args.no_amp} l1_cap={args.l1_max_min}min "
          f"adv_cap={args.adv_max_min}min budget={args.global_budget_min or 'none'}min",
          flush=True)

    es = ["--early-stop", "--early-stop-mae", "0.008", "--early-stop-hit5", "0.93",
          "--early-stop-patience", "2"]
    verdicts: list[dict] = []
    for stem in experts:
        if args.global_budget_min and elapsed_min() + args.l1_max_min + 5 > args.global_budget_min:
            print(f"SKIP {stem}: global budget nearly spent (t+{elapsed_min():.1f}min)", flush=True)
            verdicts.append({"stem": stem, "recipe": "not_run", "gate2_pass": False})
            continue

        bmp = f"{stem}.bmp"
        l1_out = out / f"{stem}_l1"
        l1_cmd = [
            sys.executable, "train_bmp_expert.py", "--data", str(data), "--bmp", bmp,
            "--out", str(l1_out), "--steps", str(args.l1_steps), "--batch", str(batch),
            "--lr", "3e-4", "--max-minutes", str(args.l1_max_min),
            "--eval-every", str(args.eval_every), "--eval-max-items", str(args.eval_max_items),
            "--checkpoint-every", "1500", "--resume", *es, *arch,
            "--progress-every", "200", "--num-workers", "2", "--device", args.device,
        ]
        if not args.no_amp:
            l1_cmd.append("--amp")
        run(f"train_{stem}_L1", l1_cmd, out)
        best = l1_out / "best.safetensors"
        if not best.exists():
            best = l1_out / "last.safetensors"
        m = eval_expert(stem, data, best, l1_out / "eval", args.device)
        best_ckpt, best_m, recipe = best, m, "L1"
        gate_key = "gate2_pass" if m.get("n_skins", 0) and m["n_skins"] > 1 else "gate1_pass"

        if not best_m.get(gate_key) and \
           (not args.global_budget_min or elapsed_min() + args.adv_max_min + 5 < args.global_budget_min):
            adv_out = out / f"{stem}_adv"
            adv_cmd = [
                sys.executable, "train_bmp_expert.py", "--data", str(data), "--bmp", bmp,
                "--out", str(adv_out), "--init-from", str(best),
                "--steps", str(args.adv_steps), "--batch", str(batch), "--lr", "1e-4",
                "--max-minutes", str(args.adv_max_min),
                "--eval-every", str(args.eval_every), "--eval-max-items", str(args.eval_max_items),
                "--checkpoint-every", "1500", "--resume",
                "--adversarial", "--adv-weight", "0.02", "--fm-weight", "1.0", "--d-lr", "2e-4",
                *es, *arch, "--progress-every", "200", "--num-workers", "2", "--device", args.device,
            ]
            run(f"train_{stem}_ADV", adv_cmd, out)
            ab = adv_out / "best.safetensors"
            if not ab.exists():
                ab = adv_out / "last.safetensors"
            am = eval_expert(stem, data, ab, adv_out / "eval", args.device)
            if (am.get(gate_key) and not best_m.get(gate_key)) or \
               (am.get("mae_mean", 9.9) < best_m.get("mae_mean", 9.9)):
                best_ckpt, best_m, recipe = ab, am, "L1+adversarial"

        dst = ck_root / stem; dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_ckpt, dst / "last.safetensors")
        v = {"stem": stem, "recipe": recipe, "gate_key": gate_key,
             "passed": bool(best_m.get(gate_key)),
             "mae_mean": round(best_m.get("mae_mean", 9.9), 5),
             "hit_5_255_mean": round(best_m.get("hit_5_255_mean", 0.0), 4),
             "n_skins": best_m.get("n_skins"), "worst_skin": best_m.get("worst_skin"),
             "fail_skins": [s for s, vv in best_m.get("per_skin", {}).items() if not vv.get("pass")]}
        verdicts.append(v)
        (out / "summary.json").write_text(json.dumps(verdicts, indent=2))  # persist as we go
        print(f"\n>>> VERDICT {stem}: {'PASS' if v['passed'] else 'FAIL'} ({v['gate_key']}) "
              f"recipe={recipe} mae={v['mae_mean']} hit5={v['hit_5_255_mean']} "
              f"worst={v['worst_skin']} fails={v['fail_skins']} (t+{elapsed_min():.1f}min)\n", flush=True)

    n_pass = sum(1 for v in verdicts if v.get("passed"))
    print("\n=== V10 GATE SWEEP DONE ===", flush=True)
    print(f"passed {n_pass}/{len(experts)} | elapsed {elapsed_min():.1f}min | ckpts in {ck_root}", flush=True)
    print(json.dumps(verdicts, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
