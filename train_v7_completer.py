#!/usr/bin/env python3
"""V7 Phase 0 trainer: asset-completion only, one skin.

Trains V7Completer on per-file partial-evidence inputs from V7CompletionDataset.

Acceptance gates (per the V7 plan):

    Gate A (one-skin):
        MAE  < 0.005
        hit5 > 0.95

Batching note: V7CompletionDataset yields per-file tensors with file-specific
spatial shapes. This trainer uses SameFileBatchSampler so every batch shares
one file_name (and thus one shape), making default DataLoader collate safe.
Pass --batch larger than 1 to exploit this; --batch 1 also works trivially.

Mixed-file batch_size > 1 with default collate is intentionally blocked.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from atlas_ai.dataset_v7_completion import V7CompletionDataset
from atlas_ai.export_spec import TRAINABLE_EXPORT_SPECS
from atlas_ai.support_mask import load_support_masks
from atlas_ai.v7_batching import SameFileBatchSampler
from models.losses_v7 import support_masked_l1_loss
from models.v7_completer import V7Completer


FILE_TO_ID: dict[str, int] = {spec.file_name: idx for idx, spec in enumerate(TRAINABLE_EXPORT_SPECS)}


def save_state_dict(path: Path, state_dict: dict) -> None:
    from safetensors.torch import save_file
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state_dict, str(path))


def parse_skin_sources(spec: str) -> dict[str, str]:
    """Parse a 'skin_id=path' list (comma-separated) or a single bare path.

    The bare-path form maps the path's basename to that directory.
    """
    out: dict[str, str] = {}
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if "=" in entry:
            sid, path = entry.split("=", 1)
            out[sid.strip()] = path.strip()
        else:
            p = Path(entry)
            out[p.name] = str(p)
    if not out:
        raise SystemExit(f"--skin-sources must declare at least one skin (got {spec!r})")
    return out


def compute_step_loss(
    model: V7Completer,
    batch: dict,
    support_masks: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    observed_rgb = batch["observed_rgb"].to(device, non_blocking=True)
    observed_mask = batch["observed_mask"].to(device, non_blocking=True)
    target_rgb = batch["target_rgb"].to(device, non_blocking=True)
    # Same-file batching guarantees every entry shares one file_name.
    file_names = batch["file_name"]
    if isinstance(file_names, (list, tuple)):
        if len({fn for fn in file_names}) != 1:
            raise RuntimeError(
                f"compute_step_loss expects a same-file batch, got {set(file_names)}"
            )
        file_name = file_names[0]
    else:
        file_name = str(file_names)
    batch_size = observed_rgb.shape[0]
    file_id = torch.full(
        (batch_size,), FILE_TO_ID[file_name], dtype=torch.long, device=device,
    )
    final_rgb = model(observed_rgb, observed_mask, file_id)
    support = support_masks[file_name].to(device=device, dtype=final_rgb.dtype)
    l1 = support_masked_l1_loss(final_rgb, target_rgb, support)
    return {"total": l1, "l1": l1.detach()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-families",
                        default="configs/state_families_classic.yaml")
    parser.add_argument("--skin-sources", required=True,
                        help="Comma-separated 'skin_id=path[,skin_id=path]' "
                             "or a single bare path; bare path uses its basename "
                             "as the skin id.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch", type=int, default=4,
                        help="Same-file batch size. Set to 1 for tiniest skins.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--file-embedding-dim", type=int, default=32)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--snapshot-every", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    skin_sources = parse_skin_sources(args.skin_sources)
    dataset = V7CompletionDataset(
        skin_sources=skin_sources,
        state_families_path=args.state_families,
        seed=args.seed,
    )
    generator = torch.Generator().manual_seed(args.seed)
    sampler = SameFileBatchSampler(
        dataset.items, batch_size=args.batch, shuffle=True, generator=generator,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and device.type == "cuda",
    )
    support_masks = load_support_masks()

    model = V7Completer(
        base_channels=args.base_channels,
        file_embedding_dim=args.file_embedding_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    config = {
        "model_version": 70,
        "args": vars(args),
        "skin_sources": skin_sources,
        "dataset_size": len(dataset),
        "batches_per_epoch": len(sampler),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))

    metrics_path = out / "metrics.jsonl"
    best = float("inf")
    step = 0
    epoch = 0
    model.train()
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        while step < args.steps:
            dataset.set_epoch(epoch)
            for batch in loader:
                if step >= args.steps:
                    break
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                    losses = compute_step_loss(model, batch, support_masks, device)
                if args.amp and device.type == "cuda":
                    scaler.scale(losses["total"]).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    losses["total"].backward()
                    optimizer.step()
                logged = {k: float(v.detach().cpu()) for k, v in losses.items()}
                logged["step"] = step
                logged["epoch"] = epoch
                logged["file_name"] = batch["file_name"][0] if isinstance(batch["file_name"], list) else batch["file_name"]
                metrics_file.write(json.dumps(logged) + "\n")
                if logged["total"] < best:
                    best = logged["total"]
                    save_state_dict(out / "best.safetensors", model.state_dict())
                if (step + 1) % args.checkpoint_every == 0 or step == 0:
                    save_state_dict(out / "last.safetensors", model.state_dict())
                if args.snapshot_every > 0 and (step + 1) % args.snapshot_every == 0:
                    save_state_dict(
                        out / f"snapshot_step{step + 1:06d}.safetensors",
                        model.state_dict(),
                    )
                step += 1
            epoch += 1
    save_state_dict(out / "last.safetensors", model.state_dict())
    print(f"trained V7Completer for {step} step(s); best step loss {best:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
