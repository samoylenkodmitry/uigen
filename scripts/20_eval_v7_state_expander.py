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
    return {
        "num_families": g("num_families_buffer"),
        "max_frames": g("max_frames_buffer"),
        "base_channels": g("base_channels_buffer"),
        "file_embedding_dim": g("file_embedding_dim_buffer"),
        "family_embedding_dim": g("family_embedding_dim_buffer"),
        "frame_embedding_dim": g("frame_embedding_dim_buffer"),
        "num_skins": g("num_skins_buffer"),
        "skin_embedding_dim": g("skin_embedding_dim_buffer"),
    }


def _per_item_metrics(pred, target, support):
    """Return per-item (mae, hit5, n_support) over support pixels."""
    sup = support.to(pred.dtype)  # [B,1,H,W]
    sup3 = sup.expand_as(pred)
    n3 = sup3.flatten(1).sum(1).clamp_min(1.0)
    mae = ((pred - target).abs() * sup3).flatten(1).sum(1) / n3
    diff255 = (pred - target).abs() * 255.0
    hit = (diff255 <= 5.0).all(dim=1, keepdim=True).to(pred.dtype)
    n1 = sup.flatten(1).sum(1).clamp_min(1.0)
    hit5 = (hit * sup).flatten(1).sum(1) / n1
    return mae, hit5, sup.flatten(1).sum(1)


def evaluate(model, loader, device) -> dict:
    model.eval()
    # accumulators[(scope, key)] = {"mae_num","hit_num","den","n"}; scope split by identity.
    fam = {"all": defaultdict(lambda: [0.0, 0.0, 0.0, 0]),
           "off": defaultdict(lambda: [0.0, 0.0, 0.0, 0])}
    skin = {"all": defaultdict(lambda: [0.0, 0.0, 0.0, 0]),
            "off": defaultdict(lambda: [0.0, 0.0, 0.0, 0])}
    with torch.no_grad():
        for batch in loader:
            source = batch["source_rgb"].to(device)
            target = batch["target_rgb"].to(device)
            support = batch["target_support"].to(device)
            skin_id = None
            if model.num_skins > 0:
                skin_id = batch["skin_index"].to(device=device, dtype=torch.long)
            pred = model(source, batch["source_idx"].to(device), batch["target_idx"].to(device),
                         batch["family_id"].to(device), batch["file_id"].to(device), skin_id=skin_id)
            mae, hit5, n = _per_item_metrics(pred, target, support)
            fams = batch["family_key"]  # globally unique (file/family)
            sids = batch["skin_id"]
            ident = batch["is_identity"].tolist()
            for i in range(pred.shape[0]):
                ni = float(n[i]); mi = float(mae[i]); hi = float(hit5[i])
                for scope in (["all"] + (["off"] if not ident[i] else [])):
                    for acc, key in ((fam, fams[i]), (skin, sids[i])):
                        s = acc[scope][key]
                        s[0] += mi * ni; s[1] += hi * ni; s[2] += ni; s[3] += 1
    def finalize(acc):
        out = {}
        for scope in ("all", "off"):
            out[scope] = {}
            for key, s in sorted(acc[scope].items()):
                den = max(s[2], 1.0)
                out[scope][key] = {"mae": s[0] / den, "hit5": s[1] / den, "pairs": s[3]}
        return out
    per_family = finalize(fam)
    per_skin = finalize(skin)

    def agg(per, scope):
        rows = per[scope]
        if not rows:
            return {"mae_median": float("nan"), "hit5_median": float("nan")}
        maes = [v["mae"] for v in rows.values()]
        hits = [v["hit5"] for v in rows.values()]
        worst_mae_fam = max(rows, key=lambda k: rows[k]["mae"])
        worst_hit_fam = min(rows, key=lambda k: rows[k]["hit5"])
        return {
            "mae_median": stats.median(maes), "hit5_median": stats.median(hits),
            "mae_mean": sum(maes) / len(maes), "hit5_mean": sum(hits) / len(hits),
            "worst_family_mae": rows[worst_mae_fam]["mae"], "worst_family_mae_name": worst_mae_fam,
            "worst_family_hit5": rows[worst_hit_fam]["hit5"], "worst_family_hit5_name": worst_hit_fam,
            "num_families": len(rows),
        }
    aggregate = {
        "off_diagonal": agg(per_family, "off"),
        "all_pairs": agg(per_family, "all"),
    }
    return {"aggregate": aggregate, "per_family": per_family, "per_skin": per_skin}


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
    n_skins = len(ds.skin_ids)
    off = result["aggregate"]["off_diagonal"]
    if n_skins == 1:
        result["gate"] = {
            "name": "S1", "off_mae": off["mae_mean"], "off_hit5": off["hit5_mean"],
            "pass": bool(off["mae_mean"] < 0.01 and off["hit5_mean"] > 0.95),
        }
    else:
        result["gate"] = {
            "name": "S2", "off_mae_median": off["mae_median"], "off_hit5_median": off["hit5_median"],
            "pass": bool(off["mae_median"] < 0.015 and off["hit5_median"] > 0.90),
        }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(text + "\n", encoding="utf-8")
    g = result["gate"]
    print(f"\nGATE {g['name']}: {'PASS' if g['pass'] else 'FAIL'}  {json.dumps({k:v for k,v in g.items() if k not in ('name','pass')})}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
