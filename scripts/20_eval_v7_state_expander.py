#!/usr/bin/env python3
"""Eval the V7.1 StateFamilyExpander: per-family / per-skin MAE + hit5.

Reports metrics over the target rect's supported pixels, split into:
  - all pairs (includes identity i==i)
  - off_diagonal (i != j) — the real state-transition task; gates use this so
    trivial identity copies don't inflate the score.

Gate S1 (one skin, all alternatives families):
    off_diagonal MAE < 0.01 and hit5 > 0.95
Gate S2 (14/16 skins):
    median-over-families off_diagonal MAE < 0.015 and hit5 > 0.90
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from atlas_ai.state_pairs_dataset import StatePairsDataset
from atlas_ai.v7_batching import SameKeyBatchSampler
from models.v7_state_expander import V7StateExpander


def parse_skin_sources(spec: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if "=" in entry:
            sid, path = entry.split("=", 1)
            out[sid.strip()] = path.strip()
        else:
            out[Path(entry).name] = str(entry)
    return out


def load_state_dict(path: Path) -> dict:
    from safetensors.torch import load_file
    return load_file(str(path))


def _model_kwargs(state: dict) -> dict:
    ver = int(state["model_version"].reshape(-1)[0].item())
    if ver != 72:
        raise SystemExit(f"need a V7StateExpander checkpoint (version 72), got {ver}")
    g = lambda k: int(state[k].reshape(-1)[0].item())
    from models.v7_state_expander import _OUTPUT_MODE_BY_CODE
    # output_mode_buffer absent on the earliest v72 checkpoints -> default residual.
    output_mode = _OUTPUT_MODE_BY_CODE.get(g("output_mode_buffer"), "residual") \
        if "output_mode_buffer" in state else "residual"
    return {
        "num_families": g("num_families_buffer"),
        "max_frames": g("max_frames_buffer"),
        "base_channels": g("base_channels_buffer"),
        "file_embedding_dim": g("file_embedding_dim_buffer"),
        "family_embedding_dim": g("family_embedding_dim_buffer"),
        "frame_embedding_dim": g("frame_embedding_dim_buffer"),
        "num_skins": g("num_skins_buffer"),
        "skin_embedding_dim": g("skin_embedding_dim_buffer"),
        "output_mode": output_mode,
    }


CHANGED_THRESHOLD = 5.0 / 255.0  # a pixel "changed" between source/target


def _ratio(num: float, den: float) -> float:
    return num / den if den > 0 else float("nan")


def _region_terms(pred, target, region_b1):
    """Batch-summed (mae_num, mae_den, hit_num, hit_den) over a [B,1,H,W] region."""
    r = region_b1.to(pred.dtype)
    r3 = r.expand_as(pred)
    mae_num = float(((pred - target).abs() * r3).sum())
    mae_den = float(r3.sum())
    hit = ((pred - target).abs() * 255.0 <= 5.0).all(dim=1, keepdim=True).to(pred.dtype)
    hit_num = float((hit * r).sum())
    hit_den = float(r.sum())
    return mae_num, mae_den, hit_num, hit_den


def _p90(vals: list[float]) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    return s[min(len(s) - 1, int(0.9 * len(s)))]


def _blank_fam():
    return {"sup": [0.0] * 4, "chg": [0.0] * 4, "unc": [0.0] * 4,
            "gate_chg_sum": 0.0, "gate_chg_n": 0.0, "gate_unc_sum": 0.0, "gate_unc_n": 0.0,
            "gate_chg_vals": [], "gate_unc_vals": [], "pairs": 0}


def evaluate(model, loader, device, *, p90_cap: int = 200000) -> dict:
    """Region-split eval over OFF-DIAGONAL (transition) pairs only.

    For state expansion most pixels are unchanged between source and target, so
    a full-support hit5 can look high while the changed region fails. We split
    every metric into support / changed / unchanged, where
        changed   = (|source-target|.amax(ch) > 5/255) & support
        unchanged = ~changed & support
    and (for gated models) report how far the gate opens on changed vs
    unchanged pixels.
    """
    model.eval()
    gated = getattr(model, "output_mode", "residual") == "gated"
    fam: dict[str, dict] = {}
    skin: dict[str, dict] = {}

    def accum(pred, target, sup_b, chg, unc, gate, fkeys, skeys):
        for fkey in set(fkeys):
            m = torch.tensor([f == fkey for f in fkeys], device=pred.device)
            a = fam.setdefault(fkey, _blank_fam())
            for region, name in ((sup_b[m], "sup"), (chg[m], "chg"), (unc[m], "unc")):
                t = _region_terms(pred[m], target[m], region)
                for k in range(4):
                    a[name][k] += t[k]
            a["pairs"] += int(m.sum())
            if gate is not None:
                for region, sk, nk, vk in ((chg[m], "gate_chg_sum", "gate_chg_n", "gate_chg_vals"),
                                           (unc[m], "gate_unc_sum", "gate_unc_n", "gate_unc_vals")):
                    if region.any():
                        gv = gate[m][region]
                        a[sk] += float(gv.sum()); a[nk] += float(gv.numel())
                        if len(a[vk]) < p90_cap:
                            a[vk].extend(gv.flatten().tolist()[: p90_cap - len(a[vk])])
        for skey in set(skeys):
            m = torch.tensor([s == skey for s in skeys], device=pred.device)
            a = skin.setdefault(skey, {"sup": [0.0] * 4, "chg": [0.0] * 4, "pairs": 0})
            for region, name in ((sup_b[m], "sup"), (chg[m], "chg")):
                t = _region_terms(pred[m], target[m], region)
                for k in range(4):
                    a[name][k] += t[k]
            a["pairs"] += int(m.sum())

    with torch.no_grad():
        for batch in loader:
            keep = (batch["is_identity"] == 0).nonzero(as_tuple=True)[0]
            if keep.numel() == 0:
                continue
            sel = lambda k: batch[k][keep]
            source = sel("source_rgb").to(device)
            target = sel("target_rgb").to(device)
            support = sel("target_support").to(device)
            skin_id = sel("skin_index").to(device=device, dtype=torch.long) if model.num_skins > 0 else None
            margs = (sel("source_idx").to(device), sel("target_idx").to(device),
                     sel("family_id").to(device), sel("file_id").to(device))
            if gated:
                pred, gate, _gl = model(source, *margs, skin_id=skin_id, return_gate=True)
            else:
                pred = model(source, *margs, skin_id=skin_id); gate = None
            diff_big = (source - target).abs().amax(dim=1, keepdim=True) > CHANGED_THRESHOLD
            sup_b = support > 0.5
            changed = diff_big & sup_b
            unchanged = (~diff_big) & sup_b
            fkeys = [batch["family_key"][int(j)] for j in keep.tolist()]
            skeys = [batch["skin_id"][int(j)] for j in keep.tolist()]
            accum(pred, target, sup_b, changed, unchanged, gate, fkeys, skeys)

    def fam_metrics(a: dict) -> dict:
        out = {"num_pairs_eval": a["pairs"]}
        for nm, pfx in (("sup", "support"), ("chg", "changed"), ("unc", "unchanged")):
            r = a[nm]
            out[f"{pfx}_mae"] = _ratio(r[0], r[1])
            out[f"{pfx}_hit5"] = _ratio(r[2], r[3])
        if a["gate_chg_n"] or a["gate_unc_n"]:
            gc, gu = _ratio(a["gate_chg_sum"], a["gate_chg_n"]), _ratio(a["gate_unc_sum"], a["gate_unc_n"])
            out["gate_changed"] = gc
            out["gate_unchanged"] = gu
            out["gate_gap"] = (gc - gu) if (gc == gc and gu == gu) else float("nan")
            out["gate_p90_changed"] = _p90(a["gate_chg_vals"])
            out["gate_p90_unchanged"] = _p90(a["gate_unc_vals"])
        return out

    per_family = {k: fam_metrics(v) for k, v in sorted(fam.items())}
    per_skin = {
        k: {"support_mae": _ratio(v["sup"][0], v["sup"][1]),
            "support_hit5": _ratio(v["sup"][2], v["sup"][3]),
            "changed_mae": _ratio(v["chg"][0], v["chg"][1]),
            "changed_hit5": _ratio(v["chg"][2], v["chg"][3]),
            "num_pairs_eval": v["pairs"]}
        for k, v in sorted(skin.items())
    }

    def col(key, only_finite=False):
        vals = [v[key] for v in per_family.values() if key in v]
        if only_finite:
            vals = [x for x in vals if x == x]
        return vals
    sup_h = col("support_hit5")
    chg_h = col("changed_hit5", only_finite=True)
    unc_h = col("unchanged_hit5", only_finite=True)
    gaps = col("gate_gap", only_finite=True)
    aggregate = {
        "num_families": len(per_family),
        "mean_support_hit5": (sum(sup_h) / len(sup_h)) if sup_h else float("nan"),
        "median_support_hit5": stats.median(sup_h) if sup_h else float("nan"),
        "min_family_support_hit5": min(sup_h) if sup_h else float("nan"),
        "mean_changed_hit5": (sum(chg_h) / len(chg_h)) if chg_h else float("nan"),
        "median_changed_hit5": stats.median(chg_h) if chg_h else float("nan"),
        "min_family_changed_hit5": min(chg_h) if chg_h else float("nan"),
        "mean_unchanged_hit5": (sum(unc_h) / len(unc_h)) if unc_h else float("nan"),
        "min_family_unchanged_hit5": min(unc_h) if unc_h else float("nan"),
        "mean_gate_gap": (sum(gaps) / len(gaps)) if gaps else float("nan"),
    }
    return {"aggregate": aggregate, "per_family": per_family, "per_skin": per_skin}


def s1_verdict(result: dict) -> dict:
    """Region-aware S1 pass/fail. Families with no changed pixels (degenerate
    transitions in this skin) are excluded from the changed-region criteria."""
    pf = result["per_family"]
    ag = result["aggregate"]
    support_ok = all(v["support_hit5"] > 0.95 for v in pf.values())
    changed_finite = [v["changed_hit5"] for v in pf.values() if v["changed_hit5"] == v["changed_hit5"]]
    unchanged_finite = [v["unchanged_hit5"] for v in pf.values() if v["unchanged_hit5"] == v["unchanged_hit5"]]
    changed_min = min(changed_finite) if changed_finite else float("nan")
    unchanged_min = min(unchanged_finite) if unchanged_finite else float("nan")
    mean_changed = ag["mean_changed_hit5"]
    gap = ag["mean_gate_gap"]
    hard = bool(support_ok and changed_min > 0.85 and mean_changed > 0.90
                and unchanged_min > 0.98 and (gap > 0.35 if gap == gap else False))
    soft = bool(support_ok and changed_min > 0.80)
    return {
        "name": "S1", "hard_pass": hard, "soft_pass": soft, "pass": hard or soft,
        "support_ok_all": support_ok, "changed_hit5_min": changed_min,
        "mean_changed_hit5": mean_changed, "unchanged_hit5_min": unchanged_min,
        "mean_gate_gap": gap,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state-families", default="configs/state_families_classic.yaml")
    ap.add_argument("--skin-sources", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    ds = StatePairsDataset(parse_skin_sources(args.skin_sources), args.state_families,
                           include_identity=True)
    sampler = SameKeyBatchSampler(ds.group_keys, batch_size=args.batch, shuffle=False)
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=0, collate_fn=default_collate)
    state = load_state_dict(Path(args.checkpoint))
    model = V7StateExpander(**_model_kwargs(state)).to(device)
    model.load_state_dict(state)

    result = evaluate(model, loader, device)
    # Inject per-family frame counts.
    nframes = {f.key: f.num_frames for f in ds.alt_families}
    for k, v in result["per_family"].items():
        v["num_frames"] = nframes.get(k)
    result["num_skins"] = len(ds.skin_ids)
    result["gate"] = s1_verdict(result)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(text + "\n", encoding="utf-8")
    g = result["gate"]
    verdict = "HARD PASS" if g["hard_pass"] else ("SOFT PASS" if g["soft_pass"] else "FAIL")
    print(f"\nGATE S1: {verdict}  support_ok={g['support_ok_all']} "
          f"changed_hit5_min={g['changed_hit5_min']:.3f} mean_changed_hit5={g['mean_changed_hit5']:.3f} "
          f"unchanged_hit5_min={g['unchanged_hit5_min']:.3f} mean_gate_gap={g['mean_gate_gap']:.3f}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
