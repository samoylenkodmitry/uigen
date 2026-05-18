#!/usr/bin/env python3
"""Train SlotNetV3.4.

V3.4 has one training contract:

    input rendered PNG -> predicted RGB atlas PNG -> expected RGB atlas PNG

No prior atlas, observed auxiliary head, dynamic masks, special-color head, or
per-slot loss weights participate in training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from atlas_ai.dataset import RenderDataset, image_to_tensor
from atlas_ai.profiles import load_atlas_profile
from models.losses import simple_atlas_loss
from models.slotnet_v34 import SlotNetV34


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


class AtlasPairDataset(Dataset):
    def __init__(self, csv_path: str | Path):
        self.rows = RenderDataset(csv_path).rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        return {
            "view": image_to_tensor(row.view_png, "RGB"),
            "target_rgb": image_to_tensor(row.atlas_png, "RGB"),
        }


def collate_atlas_pairs(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "view": torch.stack([item["view"] for item in batch], dim=0),
        "target_rgb": torch.stack([item["target_rgb"] for item in batch], dim=0),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data_v34/train.csv")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atlas-profile", default="configs/atlas_train_v1.json")
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--out", default="runs/slotnet_v34")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-every", type=int, default=2000)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--snapshot-every", type=int, default=0)
    args = parser.parse_args()

    set_seeds(args.seed)
    dataset = AtlasPairDataset(args.train)
    if args.limit_rows is not None:
        dataset.rows = dataset.rows[: args.limit_rows]
    if len(dataset) == 0:
        raise SystemExit(f"no training samples in {args.train}")

    device = torch.device(args.device)
    atlas_profile = load_atlas_profile(args.atlas_profile)
    model = SlotNetV34(atlas_profile=atlas_profile, base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        collate_fn=collate_atlas_pairs,
        num_workers=0,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(vars(args)), encoding="utf-8")
    metrics_path = out / "metrics.jsonl"

    best = float("inf")
    step = 0
    metric: dict[str, float] = {}
    try:
        while step < args.steps:
            for batch in loader:
                view = batch["view"].to(device)
                target_rgb = batch["target_rgb"].to(device)
                prediction = model(view)["prediction"]
                losses = simple_atlas_loss(prediction, target_rgb)

                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                metric = {k: float(v.detach().cpu()) for k, v in losses.items()}
                metric["step"] = step
                with metrics_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(metric, sort_keys=True) + "\n")
                if metric["total"] < best:
                    best = metric["total"]
                    save_state_dict(out / "best.safetensors", model.state_dict())
                if (step + 1) % args.checkpoint_every == 0:
                    save_state_dict(out / "last.safetensors", model.state_dict())
                if args.snapshot_every > 0 and (step + 1) % args.snapshot_every == 0:
                    save_state_dict(out / f"snapshot_step{step + 1:06d}.safetensors", model.state_dict())
                step += 1
                if step >= args.steps:
                    break
    except KeyboardInterrupt:
        print("interrupted; saving last")
    save_state_dict(out / "last.safetensors", model.state_dict())
    print(f"trained SlotNetV3.4 for {step} step(s); last loss {metric.get('total', float('nan')):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
