#!/usr/bin/env python3
"""V7 Phase 0 evaluation: per-file + aggregate MAE/hit5/sobel.

Loads a V7Completer checkpoint and reports the Phase 0 acceptance metrics
over the same dataset used for training (or a different skin set passed via
--skin-sources).

Stage A acceptance (one-skin):
    supported_mae < 0.005
    hit5        > 0.95

The eval sweeps every (skin, file) item for multiple deterministic mask rounds.
Mask distribution does not affect the target pixels, but it does affect how
partial the completion input is, so single-mask evals are too noisy for gates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from atlas_ai.dataset_v7_completion import V7CompletionDataset
from atlas_ai.export_spec import TRAINABLE_EXPORT_SPECS
from atlas_ai.support_mask import load_support_masks
from atlas_ai.v7_batching import SameFileBatchSampler
from atlas_ai.v7_masks import V7MaskWeights
from models.losses_v7 import (
    support_masked_hit5,
    support_masked_l1_loss,
    support_masked_sobel_mae,
)
from models.v7_completer import V7Completer


FILE_TO_ID: dict[str, int] = {spec.file_name: idx for idx, spec in enumerate(TRAINABLE_EXPORT_SPECS)}


def load_state_dict(path: Path) -> dict:
    from safetensors.torch import load_file
    return load_file(str(path))


def _detect_v7_kwargs(state: dict) -> dict:
    version = int(state["model_version"].reshape(-1)[0].item())
    if version != 70:
        raise SystemExit(f"need a V7 completer checkpoint (version 70), got {version}")
    return {
        "base_channels": int(state["base_channels_buffer"].reshape(-1)[0].item()),
        "file_embedding_dim": int(state["file_embedding_dim_buffer"].reshape(-1)[0].item()),
    }


def _sample_metrics(
    final_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    support: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-sample (mae, hit5, sobel_mae) on support pixels."""
    b = final_rgb.shape[0]
    mask = support
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(0)
    mask = mask.to(device=final_rgb.device, dtype=final_rgb.dtype)
    mask3 = mask.expand_as(final_rgb)
    denom3 = mask3.flatten(1).sum(dim=1).clamp_min(1.0)

    mae = ((final_rgb - target_rgb).abs() * mask3).flatten(1).sum(dim=1) / denom3

    diff_255 = (final_rgb - target_rgb).abs() * 255.0
    hit_map = (diff_255 <= 5.0).all(dim=1, keepdim=True).to(final_rgb.dtype)
    mask1 = mask.expand_as(hit_map)
    denom1 = mask1.flatten(1).sum(dim=1).clamp_min(1.0)
    hit5 = (hit_map * mask1).flatten(1).sum(dim=1) / denom1

    sobels = []
    for i in range(b):
        sobels.append(support_masked_sobel_mae(
            final_rgb[i:i + 1], target_rgb[i:i + 1], support,
        ))
    sobel = torch.stack(sobels)
    return mae, hit5, sobel


def evaluate(
    model: V7Completer,
    loader: DataLoader,
    support_masks: dict[str, torch.Tensor],
    device: torch.device,
    mask_samples: int,
) -> dict:
    model.eval()
    # Running per-file accumulators: (sum_diff, sum_hit, sum_sobel_diff, n_supported_pixels).
    per_file_mae: dict[str, float] = {s.file_name: 0.0 for s in TRAINABLE_EXPORT_SPECS}
    per_file_hit: dict[str, float] = {s.file_name: 0.0 for s in TRAINABLE_EXPORT_SPECS}
    per_file_sobel: dict[str, float] = {s.file_name: 0.0 for s in TRAINABLE_EXPORT_SPECS}
    per_file_n: dict[str, int] = {s.file_name: 0 for s in TRAINABLE_EXPORT_SPECS}
    per_file_count: dict[str, int] = {s.file_name: 0 for s in TRAINABLE_EXPORT_SPECS}
    # Per-skin accumulators (resolved at first sample because the dataset may
    # contain any subset of skin ids).
    per_skin_mae: dict[str, float] = {}
    per_skin_hit: dict[str, float] = {}
    per_skin_sobel: dict[str, float] = {}
    per_skin_n: dict[str, int] = {}
    per_skin_count: dict[str, int] = {}

    if mask_samples < 1:
        raise ValueError(f"mask_samples must be >= 1, got {mask_samples}")

    dataset = loader.dataset
    with torch.no_grad():
        for mask_round in range(mask_samples):
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(mask_round)
            for batch in loader:
                file_names = batch["file_name"]
                file_name = file_names[0] if isinstance(file_names, list) else file_names
                skin_ids = batch["skin_id"]
                if not isinstance(skin_ids, list):
                    skin_ids = [skin_ids]
                observed_rgb = batch["observed_rgb"].to(device)
                observed_mask = batch["observed_mask"].to(device)
                target_rgb = batch["target_rgb"].to(device)
                b = observed_rgb.shape[0]
                file_id = torch.full((b,), FILE_TO_ID[file_name], dtype=torch.long, device=device)
                final_rgb = model(observed_rgb, observed_mask, file_id)
                support = support_masks[file_name].to(device=device, dtype=final_rgb.dtype)
                mae_by_sample, hit_by_sample, sob_by_sample = _sample_metrics(
                    final_rgb, target_rgb, support,
                )
                mae = float(mae_by_sample.mean().item())
                hit = float(hit_by_sample.mean().item())
                sob = float(sob_by_sample.mean().item())
                n_support = int(support.bool().sum().item())
                # Accumulate weighted by per-sample support count.
                per_file_mae[file_name] += mae * n_support * b
                per_file_hit[file_name] += hit * n_support * b
                per_file_sobel[file_name] += sob * n_support * b
                per_file_n[file_name] += n_support * b
                per_file_count[file_name] += b
                # Per-skin accumulators. Items in a batch may come from
                # different skins (the sampler groups by file, not by
                # skin), so credit each item to its own skin_id.
                for i, sid in enumerate(skin_ids):
                    sid = str(sid)
                    if sid not in per_skin_n:
                        per_skin_mae[sid] = 0.0
                        per_skin_hit[sid] = 0.0
                        per_skin_sobel[sid] = 0.0
                        per_skin_n[sid] = 0
                        per_skin_count[sid] = 0
                    per_skin_mae[sid] += float(mae_by_sample[i].item()) * n_support
                    per_skin_hit[sid] += float(hit_by_sample[i].item()) * n_support
                    per_skin_sobel[sid] += float(sob_by_sample[i].item()) * n_support
                    per_skin_n[sid] += n_support
                    per_skin_count[sid] += 1

    per_file: dict[str, dict[str, float]] = {}
    agg_mae_num = agg_mae_den = 0.0
    agg_hit_num = agg_hit_den = 0.0
    agg_sob_num = 0.0
    for spec in TRAINABLE_EXPORT_SPECS:
        n = per_file_n[spec.file_name]
        if n == 0:
            per_file[spec.file_name] = {
                "supported_mae": float("nan"), "hit5": float("nan"), "sobel_mae": float("nan"),
                "samples": 0,
            }
            continue
        per_file[spec.file_name] = {
            "supported_mae": per_file_mae[spec.file_name] / n,
            "hit5":        per_file_hit[spec.file_name] / n,
            "sobel_mae":   per_file_sobel[spec.file_name] / n,
            "samples":     per_file_count[spec.file_name],
        }
        agg_mae_num += per_file_mae[spec.file_name]
        agg_mae_den += n
        agg_hit_num += per_file_hit[spec.file_name]
        agg_hit_den += n
        agg_sob_num += per_file_sobel[spec.file_name]
    aggregate = {
        "supported_mae": agg_mae_num / max(1.0, agg_mae_den),
        "hit5":        agg_hit_num / max(1.0, agg_hit_den),
        "sobel_mae":   agg_sob_num / max(1.0, agg_mae_den),
        "mask_samples": mask_samples,
    }
    per_skin: dict[str, dict[str, float]] = {}
    for sid in sorted(per_skin_n.keys()):
        n = per_skin_n[sid]
        if n == 0:
            per_skin[sid] = {
                "supported_mae": float("nan"), "hit5": float("nan"), "sobel_mae": float("nan"),
                "samples": per_skin_count[sid],
            }
            continue
        per_skin[sid] = {
            "supported_mae": per_skin_mae[sid] / n,
            "hit5":        per_skin_hit[sid] / n,
            "sobel_mae":   per_skin_sobel[sid] / n,
            "samples":     per_skin_count[sid],
        }
    return {"aggregate": aggregate, "per_file": per_file, "per_skin": per_skin}


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
            p = Path(entry)
            out[p.name] = str(p)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-families", default="configs/state_families_classic.yaml")
    parser.add_argument("--skin-sources", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mask-samples", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--mask-provenance", type=float, default=0.40)
    parser.add_argument("--mask-state-family", type=float, default=0.35)
    parser.add_argument("--mask-random-rect", type=float, default=0.20)
    parser.add_argument("--mask-whole-file", type=float, default=0.05)
    parser.add_argument("--mask-passthrough", type=float, default=0.0)
    args = parser.parse_args()

    device = torch.device(args.device)
    skin_sources = parse_skin_sources(args.skin_sources)
    dataset = V7CompletionDataset(
        skin_sources=skin_sources,
        state_families_path=args.state_families,
        mask_weights=V7MaskWeights(
            provenance=args.mask_provenance,
            state_family=args.mask_state_family,
            random_rect=args.mask_random_rect,
            whole_file=args.mask_whole_file,
            passthrough=args.mask_passthrough,
        ),
        seed=args.seed,
    )
    sampler = SameFileBatchSampler(
        dataset.items, batch_size=args.batch, shuffle=False,
    )
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
    state = load_state_dict(Path(args.checkpoint))
    kwargs = _detect_v7_kwargs(state)
    model = V7Completer(**kwargs).to(device)
    model.load_state_dict(state)
    support_masks = load_support_masks()

    result = evaluate(model, loader, support_masks, device, args.mask_samples)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
