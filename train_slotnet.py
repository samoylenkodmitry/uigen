#!/usr/bin/env python3
"""Train SlotNetV3.1.

Patches over V3:
- input-conditioned default-atlas prior + residual prediction
- atlas-layout conditioning maps injected at every decoder stage
- hidden-pixel loss weight 0.03 (was 0.25)
- sobel weight 2.0 (was 0.5)
- downsampled-atlas contrastive loss (active when batch >= 2)
- batch_size 2 by default (RTX 2070 ~ 6 GB)

GeoNet output never enters SlotNet's code path.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from atlas_ai.dataset import RenderDataset
from atlas_ai.profiles import load_atlas_profile, load_json
from models.atlas import load_full_atlas_target, pack_default_atlas_tensor
from models.losses import full_atlas_loss_v31
from models.slotnet_v31 import SlotNetV31
from models.slotnet_v32 import SlotNetV32, observed_atlas_loss


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_state_dict(path: Path, state_dict: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from safetensors.torch import save_file
        save_file(state_dict, str(path))
    except Exception:
        torch.save(state_dict, path)


def collate_with_targets(batch_rows: list, atlas_profile, magenta_policy, hidden_weight: float):
    """Custom collator: loads full atlas targets per sample, stacks to a batch."""
    out = {}
    views = []
    targets = []
    for row, view in batch_rows:
        views.append(view)
        targets.append(load_full_atlas_target(
            row.atlas_png, row.atlas_mask_png, row.visible_mask_png, row.slot_weight_f32,
            atlas_profile, magenta_policy, hidden_weight=hidden_weight,
        ))
    out["view"] = torch.stack(views, dim=0)
    keys = ["target_rgb", "atlas_mask", "visible_mask", "effective_mask", "weight_map", "special_target", "special_mask"]
    for k in keys:
        out[k] = torch.stack([t[k] for t in targets], dim=0)
    return out


class _PairedRowDataset(Dataset):
    def __init__(self, ds: RenderDataset):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds.rows[idx], self.ds[idx]["view"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data_v0/train.csv")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atlas-profile", default="configs/atlas_v1.json")
    parser.add_argument("--magenta-policy", default="configs/magenta_policy.json")
    parser.add_argument("--default-skin", default="assets/default_skin")
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--sobel-weight", type=float, default=2.0)
    parser.add_argument("--contrast-weight", type=float, default=0.05)
    parser.add_argument("--hidden-weight", type=float, default=0.03)
    parser.add_argument("--out", default="runs/slotnet_v31")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-every", type=int, default=2000)
    parser.add_argument("--limit-rows", type=int, default=None,
                        help="Use only the first N rows of train.csv. For overfit tests.")
    parser.add_argument("--alpha-start", type=float, default=1.0,
                        help="Residual alpha at step 0. Set <1.0 for warm-start schedule.")
    parser.add_argument("--alpha-end", type=float, default=1.0,
                        help="Residual alpha at --alpha-ramp-steps. Linear ramp.")
    parser.add_argument("--alpha-ramp-steps", type=int, default=10000,
                        help="Steps over which residual alpha ramps from start to end.")
    parser.add_argument("--residual-l2-start", type=float, default=0.0,
                        help="L2 penalty on RGB residual logits at step 0.")
    parser.add_argument("--residual-l2-end", type=float, default=0.0,
                        help="L2 penalty on RGB residual logits at --residual-l2-ramp-steps.")
    parser.add_argument("--residual-l2-ramp-steps", type=int, default=10000)
    parser.add_argument("--snapshot-every", type=int, default=0,
                        help="If >0, save best/last and a snapshot atlas .pt every N steps.")
    parser.add_argument("--model", default="v31", choices=["v31", "v32"],
                        help="SlotNet variant. v32 adds observed-atlas aux head.")
    parser.add_argument("--observed-weight", type=float, default=1.0,
                        help="Weight on observed_atlas auxiliary loss (V3.2 only).")
    args = parser.parse_args()

    set_seeds(args.seed)
    base_ds = RenderDataset(args.train)
    if args.limit_rows is not None:
        base_ds.rows = base_ds.rows[: args.limit_rows]
    if len(base_ds) == 0:
        raise SystemExit(f"no training samples in {args.train}")

    atlas_profile = load_atlas_profile(args.atlas_profile)
    policy = load_json(args.magenta_policy)
    device = torch.device(args.device)

    default_atlas = pack_default_atlas_tensor(args.default_skin, atlas_profile).to(device)
    if args.model == "v32":
        model = SlotNetV32(atlas_profile=atlas_profile, default_atlas=default_atlas, base_channels=args.base_channels).to(device)
    else:
        model = SlotNetV31(atlas_profile=atlas_profile, default_atlas=default_atlas, base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    paired = _PairedRowDataset(base_ds)
    loader = DataLoader(
        paired,
        batch_size=args.batch,
        shuffle=True,
        collate_fn=lambda b: collate_with_targets(b, atlas_profile, policy, args.hidden_weight),
        num_workers=0,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(vars(args)), encoding="utf-8")
    metrics_path = out / "metrics.jsonl"
    best = float("inf")
    step = 0
    metric: dict[str, float] = {}
    def alpha_at(s: int) -> float:
        if args.alpha_ramp_steps <= 0:
            return args.alpha_end
        t = min(1.0, s / max(1, args.alpha_ramp_steps))
        return args.alpha_start + (args.alpha_end - args.alpha_start) * t

    def residual_l2_at(s: int) -> float:
        if args.residual_l2_ramp_steps <= 0:
            return args.residual_l2_end
        t = min(1.0, s / max(1, args.residual_l2_ramp_steps))
        # Linear interpolation in log-space if both > 0, else linear.
        a = args.residual_l2_start
        b = args.residual_l2_end
        if a > 0 and b > 0:
            import math
            return float(math.exp(math.log(a) + (math.log(b) - math.log(a)) * t))
        return a + (b - a) * t

    try:
        while step < args.steps:
            for batch in loader:
                view = batch["view"].to(device)
                tgt = {k: v.to(device) for k, v in batch.items() if k != "view" and torch.is_tensor(v)}
                alpha = alpha_at(step)
                l2_lambda = residual_l2_at(step)
                out_pred = model(view, residual_alpha=alpha)
                losses = full_atlas_loss_v31(
                    out_pred["prediction"],
                    tgt["target_rgb"],
                    tgt["atlas_mask"],
                    tgt["effective_mask"],
                    tgt["weight_map"],
                    tgt["special_target"],
                    tgt["special_mask"],
                    sobel_weight=args.sobel_weight,
                    contrast_weight=args.contrast_weight,
                )
                # Residual L2 stabilizer (penalizes pulling RGB away from the
                # color-transferred default prior).
                if l2_lambda > 0:
                    res_l2 = out_pred["residual_logits"][:, :3].pow(2).mean()
                    total = losses["total"] + l2_lambda * res_l2
                    losses = dict(losses)
                    losses["total"] = total
                    losses["residual_l2"] = res_l2.detach()
                # Observed-atlas auxiliary head (V3.2 only).
                if args.model == "v32" and "observed_logits" in out_pred:
                    obs = observed_atlas_loss(
                        out_pred["observed_logits"],
                        tgt["target_rgb"],
                        tgt["visible_mask"],
                        tgt["atlas_mask"],
                    )
                    total = losses["total"] + args.observed_weight * obs["total"]
                    losses = dict(losses)
                    losses["total"] = total
                    losses["obs_rgb"] = obs["rgb_l1"].detach()
                    losses["obs_mask"] = obs["mask_bce"].detach()
                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                metric = {k: float(v.detach().cpu()) for k, v in losses.items()}
                metric["step"] = step
                metric["alpha"] = float(alpha)
                metric["residual_l2_lambda"] = float(l2_lambda)
                with metrics_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(metric, sort_keys=True) + "\n")
                if metric["total"] < best:
                    best = metric["total"]
                    save_state_dict(out / "best.safetensors", model.state_dict())
                if (step + 1) % args.checkpoint_every == 0:
                    save_state_dict(out / "last.safetensors", model.state_dict())
                if args.snapshot_every > 0 and (step + 1) % args.snapshot_every == 0:
                    save_state_dict(out / f"snapshot_step{step+1:06d}.safetensors", model.state_dict())
                step += 1
                if step >= args.steps:
                    break
    except KeyboardInterrupt:
        print("interrupted; saving last")
    save_state_dict(out / "last.safetensors", model.state_dict())
    print(f"trained SlotNetV3.1 for {step} step(s); last loss {metric.get('total', float('nan')):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
