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
from models.geonet80 import GeoNet80, build_geonet_targets, geonet_loss


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data_v0/train.csv")
    parser.add_argument("--val", default=None)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--fpn-channels", type=int, default=64)
    parser.add_argument("--out", default="runs/geonet80_v0")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seeds(args.seed)
    dataset = RenderDataset(args.train)
    if len(dataset) == 0:
        raise SystemExit(f"no training samples in {args.train}")
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True)
    device = torch.device(args.device)
    model = GeoNet80(base_channels=args.base_channels, fpn_channels=args.fpn_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(vars(args)), encoding="utf-8")
    metrics_path = out / "metrics.jsonl"
    best = float("inf")
    step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        for batch in loader:
            view = batch["view"].to(device)
            rects = batch["rects"].to(device)
            state = batch["state"].to(device)
            outputs = model(view, anchor_rects=rects, jitter_state_anchors=True)
            targets = build_geonet_targets(rects, outputs["heatmap"].shape[-2:])
            losses = geonet_loss(outputs, targets, state)
            (losses["total"] / args.grad_accum_steps).backward()
            if (step + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            metric = {key: float(value.detach().cpu()) for key, value in losses.items()}
            metric.update({"step": step, "epoch": epoch})
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(metric, sort_keys=True) + "\n")
            if metric["total"] < best:
                best = metric["total"]
                save_state_dict(out / "best.safetensors", model.state_dict())
            step += 1
            if step >= args.max_steps:
                save_state_dict(out / "last.safetensors", model.state_dict())
                print(f"trained GeoNet smoke for {step} step(s); last loss {metric['total']:.6f}")
                return 0
    save_state_dict(out / "last.safetensors", model.state_dict())
    print(f"trained GeoNet for {step} step(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
