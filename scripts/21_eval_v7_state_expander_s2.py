#!/usr/bin/env python3
"""V7.1 Gate S2 eval: cross-skin StateFamilyExpander generalization.

Evaluates a gated state-expander on three splits (the dataset must be built
with the SAME heldout-skins / seen-pair-fraction / seed as training):

  train_pair             skins seen, (i,j) pairs seen during training
  seen_skin_unseen_pair  skins seen, (i,j) pairs HELD OUT
  heldout_skin           skins entirely UNSEEN (the deployability test)

All metrics are region-split over transition (off-diagonal) pairs:
  support / changed / unchanged  mae + hit5,   changed = |src-tgt|.amax>5/255 & support.
Per family AND per (family, skin), so we can report min / p10 across skins and
count families below 0.85 — S2 can look fine on average but fail a few skins.

S2a pass (held-out split is what matters; oracle skin embedding NOT used):
  hard: heldout mean & median changed_hit5 > 0.90, no family mean < 0.85,
        unchanged_hit5 > 0.98, gate_gap > 0.50
  soft: heldout mean changed_hit5 > 0.85, only 1-2 families < 0.85
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
from models.v7_state_expander import _OUTPUT_MODE_BY_CODE
from models.v7_state_expander import V7StateExpander

CHANGED_THRESHOLD = 5.0 / 255.0
FILE_TYPE = {  # family file -> coarse type for the by-type rollup
    "CBUTTONS.bmp": "buttons",
    "VOLUME.bmp": "sliders", "BALANCE.bmp": "sliders",
    "SHUFREP.bmp": "toggles", "MONOSTER.bmp": "toggles", "PLAYPAUS.bmp": "toggles",
}


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


def file_type_of(family_key: str) -> str:
    f = family_key.split("/", 1)[0] + ".bmp"
    if family_key.startswith("EQMAIN/slider"):
        return "sliders"
    if f == "EQMAIN.bmp":
        return "toggles"  # on/auto buttons
    return FILE_TYPE.get(f, "other")


def _ratio(num, den):
    return num / den if den > 0 else float("nan")


def _p10(vals):
    s = sorted(vals)
    return s[min(len(s) - 1, int(0.1 * (len(s) - 1)))] if s else float("nan")


def _region_terms(pred, target, region_b1):
    r = region_b1.to(pred.dtype)
    r3 = r.expand_as(pred)
    mae_num = float(((pred - target).abs() * r3).sum()); mae_den = float(r3.sum())
    hit = ((pred - target).abs() * 255.0 <= 5.0).all(dim=1, keepdim=True).to(pred.dtype)
    return mae_num, mae_den, float((hit * r).sum()), float(r.sum())


def _model_kwargs(state):
    ver = int(state["model_version"].reshape(-1)[0].item())
    if ver not in (72, 73, 74):
        raise SystemExit(f"need a V7StateExpander checkpoint (version 72/73/74), got {ver}")
    g = lambda k: int(state[k].reshape(-1)[0].item())
    om = _OUTPUT_MODE_BY_CODE.get(g("output_mode_buffer"), "residual") if "output_mode_buffer" in state else "residual"
    return {"num_families": g("num_families_buffer"), "max_frames": g("max_frames_buffer"),
            "base_channels": g("base_channels_buffer"), "file_embedding_dim": g("file_embedding_dim_buffer"),
            "family_embedding_dim": g("family_embedding_dim_buffer"), "frame_embedding_dim": g("frame_embedding_dim_buffer"),
            "num_skins": g("num_skins_buffer"), "skin_embedding_dim": g("skin_embedding_dim_buffer"),
            "style_context_dim": g("style_context_dim_buffer") if "style_context_dim_buffer" in state else 0,
            "geometry_gate": ("geometry_gate_buffer" in state and g("geometry_gate_buffer") == 1),
            "geo_gate_hidden": g("geo_gate_hidden_buffer") if "geo_gate_hidden_buffer" in state else 64,
            "geometry_dim": g("geometry_dim_buffer") if "geometry_dim_buffer" in state else 13,
            "output_mode": om}


def evaluate(model, loader, device):
    model.eval()
    gated = getattr(model, "output_mode", "residual") == "gated"
    # acc[(split, fkey, skey)] -> region sums + gate sums
    def blank():
        return {"sup": [0.0] * 4, "chg": [0.0] * 4, "unc": [0.0] * 4,
                "gchg_s": 0.0, "gchg_n": 0.0, "gunc_s": 0.0, "gunc_n": 0.0, "pairs": 0}
    acc: dict = defaultdict(blank)
    with torch.no_grad():
        for batch in loader:
            keep = (batch["is_identity"] == 0).nonzero(as_tuple=True)[0]
            if keep.numel() == 0:
                continue
            sel = lambda k: batch[k][keep]
            source = sel("source_rgb").to(device); target = sel("target_rgb").to(device)
            support = sel("target_support").to(device)
            skin_id = sel("skin_index").to(device=device, dtype=torch.long) if model.num_skins > 0 else None
            pair_geom = sel("pair_geom").to(device) if model.geometry_gate else None
            margs = (sel("source_idx").to(device), sel("target_idx").to(device),
                     sel("family_id").to(device), sel("file_id").to(device))
            if gated:
                pred, gate, _gl = model(source, *margs, skin_id=skin_id, pair_geom=pair_geom, return_gate=True)
            else:
                pred = model(source, *margs, skin_id=skin_id); gate = None
            diff_big = (source - target).abs().amax(dim=1, keepdim=True) > CHANGED_THRESHOLD
            sup_b = support > 0.5
            changed = diff_big & sup_b
            unchanged = (~diff_big) & sup_b
            splits = [batch["split"][int(j)] for j in keep.tolist()]
            fkeys = [batch["family_key"][int(j)] for j in keep.tolist()]
            skeys = [batch["skin_id"][int(j)] for j in keep.tolist()]
            groups = {(splits[n], fkeys[n], skeys[n]) for n in range(len(splits))}
            for (sp, fk, sk) in groups:
                m = torch.tensor([splits[n] == sp and fkeys[n] == fk and skeys[n] == sk
                                  for n in range(len(splits))], device=device)
                a = acc[(sp, fk, sk)]
                for region, name in ((sup_b[m], "sup"), (changed[m], "chg"), (unchanged[m], "unc")):
                    t = _region_terms(pred[m], target[m], region)
                    for k in range(4):
                        a[name][k] += t[k]
                a["pairs"] += int(m.sum())
                if gated:
                    for region, sk_, nk in ((changed[m], "gchg_s", "gchg_n"), (unchanged[m], "gunc_s", "gunc_n")):
                        if region.any():
                            gv = gate[m][region]; a[sk_] += float(gv.sum()); a[nk] += float(gv.numel())
    return acc


def summarize(acc, split):
    """Per-family (aggregated over skins) + per-(family,skin) changed_hit5."""
    fam_sums: dict = defaultdict(lambda: {"sup": [0.0] * 4, "chg": [0.0] * 4, "unc": [0.0] * 4,
                                          "gchg_s": 0.0, "gchg_n": 0.0, "gunc_s": 0.0, "gunc_n": 0.0, "pairs": 0})
    fam_skin_chg: dict = defaultdict(list)  # fkey -> [changed_hit5 per skin]
    for (sp, fk, sk), a in acc.items():
        if sp != split:
            continue
        f = fam_sums[fk]
        for nm in ("sup", "chg", "unc"):
            for k in range(4):
                f[nm][k] += a[nm][k]
        for kk in ("gchg_s", "gchg_n", "gunc_s", "gunc_n", "pairs"):
            f[kk] += a[kk]
        if a["chg"][3] > 0:
            fam_skin_chg[fk].append(_ratio(a["chg"][2], a["chg"][3]))
    per_family = {}
    for fk, f in sorted(fam_sums.items()):
        gc, gu = _ratio(f["gchg_s"], f["gchg_n"]), _ratio(f["gunc_s"], f["gunc_n"])
        skins_chg = fam_skin_chg.get(fk, [])
        per_family[fk] = {
            "support_hit5": _ratio(f["sup"][2], f["sup"][3]),
            "changed_hit5": _ratio(f["chg"][2], f["chg"][3]),
            "changed_mae": _ratio(f["chg"][0], f["chg"][1]),
            "unchanged_hit5": _ratio(f["unc"][2], f["unc"][3]),
            "gate_changed": gc, "gate_unchanged": gu,
            "gate_gap": (gc - gu) if (gc == gc and gu == gu) else float("nan"),
            "num_pairs_eval": f["pairs"], "num_skins": len(skins_chg),
            "min_skin_changed_hit5": min(skins_chg) if skins_chg else float("nan"),
            "p10_skin_changed_hit5": _p10(skins_chg) if skins_chg else float("nan"),
            "num_skins_below_0.85": sum(1 for x in skins_chg if x < 0.85),
        }
    return per_family


def _agg(per_family):
    fams = [v for v in per_family.values() if v["changed_hit5"] == v["changed_hit5"]]
    if not fams:
        return {}
    chg = [v["changed_hit5"] for v in fams]
    sup = [v["support_hit5"] for v in fams]
    unc = [v["unchanged_hit5"] for v in fams if v["unchanged_hit5"] == v["unchanged_hit5"]]
    gaps = [v["gate_gap"] for v in fams if v["gate_gap"] == v["gate_gap"]]
    by_type = defaultdict(list)
    for fk, v in per_family.items():
        if v["changed_hit5"] == v["changed_hit5"]:
            by_type[file_type_of(fk)].append(v["changed_hit5"])
    return {
        "mean_changed_hit5": sum(chg) / len(chg), "median_changed_hit5": stats.median(chg),
        "min_family_changed_hit5": min(chg),
        "mean_support_hit5": sum(sup) / len(sup), "min_family_support_hit5": min(sup),
        "min_family_unchanged_hit5": min(unc) if unc else float("nan"),
        "mean_gate_gap": (sum(gaps) / len(gaps)) if gaps else float("nan"),
        "num_families_below_0.85_changed": sum(1 for x in chg if x < 0.85),
        "changed_hit5_by_file_type": {t: sum(v) / len(v) for t, v in sorted(by_type.items())},
    }


def s2a_verdict(heldout_per_family, heldout_agg):
    if not heldout_agg:
        return {"name": "S2a", "pass": False, "note": "no held-out split present"}
    below = [k for k, v in heldout_per_family.items()
             if v["changed_hit5"] == v["changed_hit5"] and v["changed_hit5"] < 0.85]
    hard = bool(heldout_agg["mean_changed_hit5"] > 0.90 and heldout_agg["median_changed_hit5"] > 0.90
                and heldout_agg["min_family_changed_hit5"] > 0.85
                and heldout_agg["min_family_unchanged_hit5"] > 0.98
                and (heldout_agg["mean_gate_gap"] > 0.50 if heldout_agg["mean_gate_gap"] == heldout_agg["mean_gate_gap"] else False))
    soft = bool(heldout_agg["mean_changed_hit5"] > 0.85 and len(below) <= 2)
    return {"name": "S2a", "hard_pass": hard, "soft_pass": soft, "pass": hard or soft,
            "heldout_mean_changed_hit5": heldout_agg["mean_changed_hit5"],
            "heldout_median_changed_hit5": heldout_agg["median_changed_hit5"],
            "heldout_min_family_changed_hit5": heldout_agg["min_family_changed_hit5"],
            "heldout_unchanged_min": heldout_agg["min_family_unchanged_hit5"],
            "heldout_mean_gate_gap": heldout_agg["mean_gate_gap"],
            "families_below_0.85": sorted(below)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state-families", default="configs/state_families_classic.yaml")
    ap.add_argument("--skin-sources", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--heldout-skins", required=True,
                    help="Comma-separated held-out skin ids (must match training).")
    ap.add_argument("--seen-pair-val-fraction", type=float, default=0.0)
    ap.add_argument("--local-delta", type=int, default=2)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    heldout = [s.strip() for s in args.heldout_skins.split(",") if s.strip()]
    ds = StatePairsDataset(
        parse_skin_sources(args.skin_sources), args.state_families,
        include_identity=False, heldout_skins=heldout,
        heldout_pair_fraction=args.seen_pair_val_fraction,
        local_delta=args.local_delta, split_seed=args.split_seed,
    )
    sampler = SameKeyBatchSampler(ds.group_keys, batch_size=args.batch, shuffle=False)
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=0, collate_fn=default_collate)
    from safetensors.torch import load_file
    state = load_file(args.checkpoint)
    model = V7StateExpander(**_model_kwargs(state)).to(device)
    model.load_state_dict(state)

    acc = evaluate(model, loader, device)
    result = {"splits": {}, "num_skins": len(ds.skin_ids), "heldout_skins": heldout}
    for split in ("train", "seen_val", "heldout"):
        pf = summarize(acc, split)
        if pf:
            result["splits"][split] = {"aggregate": _agg(pf), "per_family": pf}
    held = result["splits"].get("heldout", {})
    result["gate"] = s2a_verdict(held.get("per_family", {}), held.get("aggregate", {}))

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(text + "\n", encoding="utf-8")
    g = result["gate"]
    verdict = "HARD PASS" if g.get("hard_pass") else ("SOFT PASS" if g.get("soft_pass") else "FAIL")
    print(f"\nGATE S2a: {verdict}  heldout mean_changed_hit5={g.get('heldout_mean_changed_hit5', float('nan')):.3f} "
          f"min_family={g.get('heldout_min_family_changed_hit5', float('nan')):.3f} "
          f"unchanged_min={g.get('heldout_unchanged_min', float('nan')):.3f} "
          f"gate_gap={g.get('heldout_mean_gate_gap', float('nan')):.3f} "
          f"families<0.85={g.get('families_below_0.85')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
