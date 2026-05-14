#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from atlas_ai.dataset import RenderDataset
from atlas_ai.profiles import load_atlas_profile, load_json
from models.atlas import load_slot_target
from models.losses import slot_loss
from models.slotnet_v1 import SlotNetV1


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


def source_rect_for_slot(rects: torch.Tensor, slot_name: str) -> torch.Tensor:
    mapping = {
        "MAIN": [0],
        "TITLEBAR": [1, 23],
        "CBUTTONS": [6],
        "SHUFREP": [17, 18, 19, 20],
        "MONOSTER": [21],
        "PLAYPAUS": [22],
        "EQMAIN": [24],
        "PLEDIT": [42],
        "POSBAR": [5],
        "VOLUME": [13],
        "BALANCE": [15],
    }
    fallback = {"EQMAIN": 24, "PLEDIT": 42}.get(slot_name, 0)
    ids = mapping[slot_name]
    visible = rects[ids, 4] > 0
    if visible.any():
        selected = rects[ids][visible]
        x0 = selected[:, 0].min()
        y0 = selected[:, 1].min()
        x1 = selected[:, 2].max()
        y1 = selected[:, 3].max()
        w = x1 - x0
        h = y1 - y0
        expand = 0.12 if slot_name not in {"MAIN", "EQMAIN", "PLEDIT"} else 0.0
        return torch.stack(((x0 - w * expand).clamp(0, 1), (y0 - h * expand).clamp(0, 1), (x1 + w * expand).clamp(0, 1), (y1 + h * expand).clamp(0, 1), torch.ones_like(x0)))
    return rects[fallback]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data_v0/train.csv")
    parser.add_argument("--slot", default="MAIN")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atlas-profile", default="configs/atlas_v1.json")
    parser.add_argument("--magenta-policy", default="configs/magenta_policy.json")
    parser.add_argument("--out", default="runs/slotnet_v1_stage_a")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seeds(args.seed)
    dataset = RenderDataset(args.train)
    if len(dataset) == 0:
        raise SystemExit(f"no training samples in {args.train}")
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    atlas_profile = load_atlas_profile(args.atlas_profile)
    slot = atlas_profile.slots_by_name[args.slot]
    policy = load_json(args.magenta_policy)
    device = torch.device(args.device)
    model = SlotNetV1().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(vars(args)), encoding="utf-8")
    metrics_path = out / "metrics.jsonl"
    best = float("inf")
    step = 0
    while step < args.steps:
        for batch in loader:
            row_idx = step % len(dataset.rows)
            row = dataset.rows[row_idx]
            view = batch["view"].to(device)
            rect = source_rect_for_slot(batch["rects"][0], args.slot).unsqueeze(0).to(device)
            state = batch["state"].to(device)
            slot_id = torch.tensor([slot.id], dtype=torch.long, device=device)
            target = load_slot_target(row.atlas_png, row.atlas_mask_png, row.visible_mask_png, row.slot_weight_f32, args.slot, atlas_profile, policy)
            target = {k: v.to(device) if torch.is_tensor(v) else v for k, v in target.items()}
            output = model(view, rect, state, slot_id, (slot.h, slot.w))
            losses = slot_loss(
                output["prediction"],
                target["target_rgb"],
                target["effective_mask"],
                target["atlas_mask"],
                target["special_target"],
                bool(target["special_enabled"]),
                float(target["slot_weight"]),
            )
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            metric = {key: float(value.detach().cpu()) for key, value in losses.items()}
            metric.update({"step": step, "slot": args.slot})
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(metric, sort_keys=True) + "\n")
            if metric["total"] < best:
                best = metric["total"]
                save_state_dict(out / "best.safetensors", model.state_dict())
            step += 1
            if step >= args.steps:
                break
    save_state_dict(out / "last.safetensors", model.state_dict())
    print(f"trained SlotNet smoke for {step} step(s); last loss {metric['total']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
