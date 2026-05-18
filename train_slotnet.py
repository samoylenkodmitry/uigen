#!/usr/bin/env python3
"""Train SlotNetV3.5.

V3.5 has one training contract:

    input rendered PNG -> predicted exported BMP tensors -> expected BMP pixels

No prior atlas, observed auxiliary head, dynamic masks, special-color head, or
distortion side channel participates in training.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from atlas_ai.dataset import RenderDataset, image_to_tensor
from atlas_ai.export_spec import (
    TRAINABLE_EXPORT_SPECS,
    load_file_weight_overrides,
    specs_weight_map,
    with_file_weights,
)
from models.losses import exported_files_loss
from models.slotnet_v35 import SlotNetV35


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


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    try:
        from safetensors.torch import load_file
        return load_file(str(path))
    except Exception:
        return torch.load(path, map_location="cpu")


def git_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


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


def dataset_summary(dataset: AtlasPairDataset, csv_path: str | Path) -> dict:
    skin_counts = Counter(row.skin_id for row in dataset.rows)
    variant_counts = Counter(row.variant_id for row in dataset.rows)
    return {
        "csv_path": str(csv_path),
        "row_count": len(dataset.rows),
        "unique_skin_count": len(skin_counts),
        "unique_variant_count": len(variant_counts),
        "variants_per_skin_min": min(skin_counts.values()) if skin_counts else 0,
        "variants_per_skin_max": max(skin_counts.values()) if skin_counts else 0,
        "skin_row_counts": dict(sorted(skin_counts.items())),
    }


def make_loader(
    dataset: AtlasPairDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int | None,
) -> DataLoader:
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "collate_fn": collate_atlas_pairs,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **kwargs)


def move_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
    *,
    non_blocking: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        batch["view"].to(device, non_blocking=non_blocking),
        batch["target_rgb"].to(device, non_blocking=non_blocking),
    )


def tensor_metrics(losses: dict[str, torch.Tensor]) -> dict[str, float]:
    return {key: float(value.detach().cpu()) for key, value in losses.items()}


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    specs,
    edge_weight: float,
    use_amp: bool,
    pin_memory: bool,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    samples = 0
    for batch in loader:
        view, target_rgb = move_batch(batch, device, non_blocking=pin_memory)
        with torch.cuda.amp.autocast(enabled=use_amp):
            output = model(view)
            losses = exported_files_loss(
                output["files"],
                target_rgb,
                specs=specs,
                edge_weight=edge_weight,
            )
        batch_size = int(view.shape[0])
        for key, value in tensor_metrics(losses).items():
            totals[key] = totals.get(key, 0.0) + value * batch_size
        samples += batch_size
    model.train()
    return {f"val_{key}": value / max(samples, 1) for key, value in totals.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data_v35/train.csv")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--style-dim", type=int, default=192)
    parser.add_argument("--head-channels", type=int, default=None)
    parser.add_argument("--edge-weight", type=float, default=1.5)
    parser.add_argument("--file-weights-yaml", default=None)
    parser.add_argument("--out", default="runs/slotnet_v35")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-every", type=int, default=2000)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--snapshot-every", type=int, default=0)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--val-csv", default=None)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    set_seeds(args.seed)
    dataset = AtlasPairDataset(args.train)
    if args.limit_rows is not None:
        dataset.rows = dataset.rows[: args.limit_rows]
    if len(dataset) == 0:
        raise SystemExit(f"no training samples in {args.train}")

    val_dataset = AtlasPairDataset(args.val_csv) if args.val_csv else None
    if val_dataset is not None and len(val_dataset) == 0:
        raise SystemExit(f"no validation samples in {args.val_csv}")

    device = torch.device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    specs = TRAINABLE_EXPORT_SPECS
    if args.file_weights_yaml:
        specs = with_file_weights(load_file_weight_overrides(args.file_weights_yaml), specs)

    model = SlotNetV35(
        base_channels=args.base_channels,
        style_dim=args.style_dim,
        head_channels=args.head_channels,
    ).to(device)
    if args.resume:
        model.load_state_dict(load_state_dict(Path(args.resume)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    loader = make_loader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = make_loader(
            val_dataset,
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["model_version"] = 35
    config["git_commit"] = git_commit_hash()
    config["train_dataset"] = dataset_summary(dataset, args.train)
    config["val_dataset"] = dataset_summary(val_dataset, args.val_csv) if val_dataset is not None else None
    config["file_weights"] = specs_weight_map(specs)
    config["best_metric"] = "val_exported_l1" if val_loader is not None and args.eval_every > 0 else "train_total"
    (out / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    metrics_path = out / "metrics.jsonl"
    val_metrics_path = out / "val_metrics.jsonl"

    best = float("inf")
    step = 0
    metric: dict[str, float] = {}
    try:
        while step < args.steps:
            for batch in loader:
                view, target_rgb = move_batch(batch, device, non_blocking=args.pin_memory)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    output = model(view)
                    losses = exported_files_loss(
                        output["files"],
                        target_rgb,
                        specs=specs,
                        edge_weight=args.edge_weight,
                    )

                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(losses["total"]).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    losses["total"].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                metric = tensor_metrics(losses)
                metric["step"] = step
                with metrics_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(metric, sort_keys=True) + "\n")

                should_save_best = False
                best_candidate = metric["total"]
                if val_loader is not None and args.eval_every > 0 and (step + 1) % args.eval_every == 0:
                    val_metric = evaluate_model(
                        model,
                        val_loader,
                        device,
                        specs=specs,
                        edge_weight=args.edge_weight,
                        use_amp=use_amp,
                        pin_memory=args.pin_memory,
                    )
                    val_metric["step"] = step
                    with val_metrics_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(val_metric, sort_keys=True) + "\n")
                    best_candidate = val_metric["val_exported_l1"]
                    should_save_best = best_candidate < best
                elif (val_loader is None or args.eval_every <= 0) and best_candidate < best:
                    should_save_best = True

                if should_save_best:
                    best = best_candidate
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
    print(
        f"trained SlotNetV3.5 for {step} step(s); "
        f"last loss {metric.get('total', float('nan')):.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
