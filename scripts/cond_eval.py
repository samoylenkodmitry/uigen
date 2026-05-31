#!/usr/bin/env python3
"""Conditioning gate for V11 (codex): does the model produce SKIN-SPECIFIC atlases
for unseen skins, vs a generic average? Exact MAE alone is misleading for this
generative style->atlas task; these metrics are the real gate.

For one prediction per held-out skin:
  own_mae      = mae(pred_i, target_i)
  shuffled_mae = mae(pred_i, target_{i+1})        # vs a different skin's target
  gap          = shuffled_mae - own_mae           # >0 and large => conditions on input
  pred_div     = mean pairwise |pred_i - pred_j|   # high => outputs differ per skin
  tgt_div      = same for targets (ceiling)
Usage: python scripts/cond_eval.py --data data_v10n_held16 --bmp CBUTTONS.bmp --checkpoint X.safetensors
"""
from __future__ import annotations
import argparse, collections, sys
from pathlib import Path
import torch
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from safetensors.torch import load_file
from atlas_ai.dataset_v10_bmp import BMPExpertDataset
import eval_bmp_expert as E


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--bmp", required=True)
    ap.add_argument("--checkpoint", required=True); ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    ds = BMPExpertDataset(a.data, a.bmp)
    byskin = collections.OrderedDict()
    for i in range(len(ds)):
        it = ds[i]; byskin.setdefault(it["skin_id"], it)
    items = list(byskin.values())
    st = load_file(a.checkpoint); m = E._build_expert_from_state(st).to(dev).eval()
    m.load_state_dict(st, strict=False)
    P, T = [], []
    with torch.no_grad():
        for it in items:
            P.append(m(it["render"].unsqueeze(0).to(dev))[0].clamp(0, 1).cpu()); T.append(it["target"])
    P = torch.stack(P); T = torch.stack(T); n = len(P)
    own = torch.stack([(P[i] - T[i]).abs().mean() for i in range(n)]).mean().item()
    shuf = torch.stack([(P[i] - T[(i + 1) % n]).abs().mean() for i in range(n)]).mean().item()
    pdiv = torch.stack([(P[i] - P[(i + 1) % n]).abs().mean() for i in range(n)]).mean().item()
    tdiv = torch.stack([(T[i] - T[(i + 1) % n]).abs().mean() for i in range(n)]).mean().item()
    print(f"COND {a.bmp} n={n}: own={own:.4f} shuffled={shuf:.4f} gap={shuf-own:+.4f} "
          f"pred_div={pdiv:.4f} tgt_div={tdiv:.4f} div_ratio={pdiv/max(tdiv,1e-6):.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
