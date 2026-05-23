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
                observed_rgb = batch["observed_rgb"].to(device)
                observed_mask = batch["observed_mask"].to(device)
                target_rgb = batch["target_rgb"].to(device)
                b = observed_rgb.shape[0]
                file_id = torch.full((b,), FILE_TO_ID[file_name], dtype=torch.long, device=device)
                final_rgb = model(observed_rgb, observed_mask, file_id)
                support = support_masks[file_name].to(device=device, dtype=final_rgb.dtype)
                mae = float(support_masked_l1_loss(final_rgb, target_rgb, support).item())
                hit = float(support_masked_hit5(final_rgb, target_rgb, support).item())
                sob = float(support_masked_sobel_mae(final_rgb, target_rgb, support).item())
                n_support = int(support.bool().sum().item())
                # Accumulate weighted by per-sample support count.
                per_file_mae[file_name] += mae * n_support * b
                per_file_hit[file_name] += hit * n_support * b
                per_file_sobel[file_name] += sob * n_support * b
                per_file_n[file_name] += n_support * b
                per_file_count[file_name] += b

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
    return {"aggregate": aggregate, "per_file": per_file}


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
    args = parser.parse_args()

    device = torch.device(args.device)
    skin_sources = parse_skin_sources(args.skin_sources)
    dataset = V7CompletionDataset(
        skin_sources=skin_sources,
        state_families_path=args.state_families,
        mask_weights=V7MaskWeights(),
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
