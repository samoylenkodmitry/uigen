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
from atlas_ai.v7_masks import V7MaskWeights, has_alternatives, has_available_mask_mode
from models.losses_v7 import (
    hidden_supported_hit5_terms,
    hidden_supported_l1_terms,
    hidden_supported_sobel_terms,
    observed_passthrough_terms,
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
    if version not in (70, 71):
        raise SystemExit(f"need a V7 completer checkpoint (version 70 or 71), got {version}")
    kwargs = {
        "base_channels": int(state["base_channels_buffer"].reshape(-1)[0].item()),
        "file_embedding_dim": int(state["file_embedding_dim_buffer"].reshape(-1)[0].item()),
    }
    if "num_skins_buffer" in state:
        kwargs["num_skins"] = int(state["num_skins_buffer"].reshape(-1)[0].item())
    if "skin_embedding_dim_buffer" in state:
        kwargs["skin_embedding_dim"] = int(
            state["skin_embedding_dim_buffer"].reshape(-1)[0].item()
        )
    return kwargs


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


# Hidden-normalized metric keys -> their term functions. Each term function
# returns ([B] num, [B] den) so the eval can accumulate num/den across all
# batches whose hidden-pixel counts differ. See models/losses_v7.py.
HIDDEN_TERMS = {
    "hidden_supported_mae": hidden_supported_l1_terms,
    "hidden_hit5": hidden_supported_hit5_terms,
    "hidden_sobel_mae": hidden_supported_sobel_terms,
    "observed_passthrough_mae": observed_passthrough_terms,
}


def _ratio(num: float, den: float) -> float:
    return num / den if den > 0 else float("nan")


def eval_file_coverage(dataset) -> tuple[list[str], list[str]]:
    """Split the trainable files into (eligible, skipped) under the dataset's
    mask weights. A file is eligible if at least one mask mode survives its
    prerequisites: provenance needs a pool, state_family needs an
    `alternatives` family. Files with neither (POSBAR, MAIN, TITLEBAR, ...)
    are skipped for a state_family-only mix rather than crashing the sampler.
    """
    eligible: list[str] = []
    skipped: list[str] = []
    for spec in TRAINABLE_EXPORT_SPECS:
        rects = dataset.state_families.get(spec.file_name, [])
        pool = dataset.provenance_pools.get(spec.file_name)
        ok = has_available_mask_mode(
            dataset.mask_weights,
            have_provenance=bool(pool),
            have_state_family=has_alternatives(rects),
        )
        (eligible if ok else skipped).append(spec.file_name)
    return eligible, skipped


def evaluate(
    model: V7Completer,
    loader: DataLoader,
    support_masks: dict[str, torch.Tensor],
    device: torch.device,
    mask_samples: int,
) -> dict:
    model.eval()
    # Full-supported (debug/secondary) per-file accumulators.
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
    # Hidden-normalized (primary) accumulators: num/den per metric.
    per_file_hnum = {k: {s.file_name: 0.0 for s in TRAINABLE_EXPORT_SPECS} for k in HIDDEN_TERMS}
    per_file_hden = {k: {s.file_name: 0.0 for s in TRAINABLE_EXPORT_SPECS} for k in HIDDEN_TERMS}
    per_skin_hnum: dict[str, dict[str, float]] = {k: {} for k in HIDDEN_TERMS}
    per_skin_hden: dict[str, dict[str, float]] = {k: {} for k in HIDDEN_TERMS}
    # Per-mode hidden accumulators, keyed by the mask mode that produced each
    # item ("state_family" / "random_rect" / "provenance" / ...). This is what
    # lets us tell a state_family failure apart from a random_rect success.
    per_mode_hnum: dict[str, dict[str, float]] = {k: {} for k in HIDDEN_TERMS}
    per_mode_hden: dict[str, dict[str, float]] = {k: {} for k in HIDDEN_TERMS}
    per_mode_count: dict[str, int] = {}

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
                modes = batch.get("mode")
                if modes is not None and not isinstance(modes, (list, tuple)):
                    modes = [modes]
                observed_rgb = batch["observed_rgb"].to(device)
                observed_mask = batch["observed_mask"].to(device)
                target_rgb = batch["target_rgb"].to(device)
                b = observed_rgb.shape[0]
                file_id = torch.full((b,), FILE_TO_ID[file_name], dtype=torch.long, device=device)
                num_skins = getattr(model, "num_skins", 0)
                skin_index = batch.get("skin_index")
                if num_skins > 0 and skin_index is not None:
                    if not isinstance(skin_index, torch.Tensor):
                        skin_index = torch.as_tensor(skin_index, dtype=torch.long)
                    skin_id_tensor = skin_index.to(device=device, dtype=torch.long)
                    final_rgb = model(observed_rgb, observed_mask, file_id, skin_id=skin_id_tensor)
                else:
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
                # Hidden-normalized terms (per item) for this batch.
                observed_mask_f = observed_mask.to(dtype=final_rgb.dtype)
                hidden_terms = {
                    k: fn(final_rgb, target_rgb, observed_mask_f, support)
                    for k, fn in HIDDEN_TERMS.items()
                }
                for k, (num, den) in hidden_terms.items():
                    per_file_hnum[k][file_name] += float(num.sum().item())
                    per_file_hden[k][file_name] += float(den.sum().item())
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
                        for k in HIDDEN_TERMS:
                            per_skin_hnum[k][sid] = 0.0
                            per_skin_hden[k][sid] = 0.0
                    per_skin_mae[sid] += float(mae_by_sample[i].item()) * n_support
                    per_skin_hit[sid] += float(hit_by_sample[i].item()) * n_support
                    per_skin_sobel[sid] += float(sob_by_sample[i].item()) * n_support
                    per_skin_n[sid] += n_support
                    per_skin_count[sid] += 1
                    for k, (num, den) in hidden_terms.items():
                        per_skin_hnum[k][sid] += float(num[i].item())
                        per_skin_hden[k][sid] += float(den[i].item())
                    # Per-mode: credit each item to the mask mode that produced it.
                    if modes is not None and i < len(modes):
                        mode_i = str(modes[i])
                        per_mode_count[mode_i] = per_mode_count.get(mode_i, 0) + 1
                        for k, (num, den) in hidden_terms.items():
                            per_mode_hnum[k].setdefault(mode_i, 0.0)
                            per_mode_hden[k].setdefault(mode_i, 0.0)
                            per_mode_hnum[k][mode_i] += float(num[i].item())
                            per_mode_hden[k][mode_i] += float(den[i].item())

    def _hidden_block(hnum, hden, name) -> dict[str, float]:
        return {k: _ratio(hnum[k][name], hden[k][name]) for k in HIDDEN_TERMS}

    per_file: dict[str, dict[str, float]] = {}
    evaluated_files: list[str] = []
    agg_mae_num = agg_mae_den = 0.0
    agg_hit_num = agg_hit_den = 0.0
    agg_sob_num = 0.0
    agg_hnum = {k: 0.0 for k in HIDDEN_TERMS}
    agg_hden = {k: 0.0 for k in HIDDEN_TERMS}
    for spec in TRAINABLE_EXPORT_SPECS:
        name = spec.file_name
        n = per_file_n[name]
        if n == 0:
            # Not sampled this run (e.g. no eligible mask mode under the
            # requested weights). Reported in the coverage block, omitted here.
            continue
        evaluated_files.append(name)
        for k in HIDDEN_TERMS:
            agg_hnum[k] += per_file_hnum[k][name]
            agg_hden[k] += per_file_hden[k][name]
        per_file[name] = {
            "supported_mae": per_file_mae[name] / n,
            "hit5":        per_file_hit[name] / n,
            "sobel_mae":   per_file_sobel[name] / n,
            "samples":     per_file_count[name],
            **_hidden_block(per_file_hnum, per_file_hden, name),
        }
        agg_mae_num += per_file_mae[name]
        agg_mae_den += n
        agg_hit_num += per_file_hit[name]
        agg_hit_den += n
        agg_sob_num += per_file_sobel[name]
    aggregate = {
        "supported_mae": agg_mae_num / max(1.0, agg_mae_den),
        "hit5":        agg_hit_num / max(1.0, agg_hit_den),
        "sobel_mae":   agg_sob_num / max(1.0, agg_mae_den),
        "mask_samples": mask_samples,
        **{k: _ratio(agg_hnum[k], agg_hden[k]) for k in HIDDEN_TERMS},
    }
    per_mode: dict[str, dict[str, float]] = {}
    for mode_name in sorted(per_mode_count):
        per_mode[mode_name] = {
            "samples": per_mode_count[mode_name],
            **{k: _ratio(per_mode_hnum[k].get(mode_name, 0.0),
                         per_mode_hden[k].get(mode_name, 0.0)) for k in HIDDEN_TERMS},
        }
    all_files = [s.file_name for s in TRAINABLE_EXPORT_SPECS]
    coverage = {
        "evaluated_files": evaluated_files,
        "skipped_files": [f for f in all_files if f not in evaluated_files],
    }
    per_skin: dict[str, dict[str, float]] = {}
    for sid in sorted(per_skin_n.keys()):
        n = per_skin_n[sid]
        if n == 0:
            per_skin[sid] = {
                "supported_mae": float("nan"), "hit5": float("nan"), "sobel_mae": float("nan"),
                "samples": per_skin_count[sid],
                **_hidden_block(per_skin_hnum, per_skin_hden, sid),
            }
            continue
        per_skin[sid] = {
            "supported_mae": per_skin_mae[sid] / n,
            "hit5":        per_skin_hit[sid] / n,
            "sobel_mae":   per_skin_sobel[sid] / n,
            "samples":     per_skin_count[sid],
            **_hidden_block(per_skin_hnum, per_skin_hden, sid),
        }
    return {
        "aggregate": aggregate,
        "per_file": per_file,
        "per_skin": per_skin,
        "per_mode": per_mode,
        "coverage": coverage,
    }


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
    eligible_files, skipped_files = eval_file_coverage(dataset)
    if not eligible_files:
        raise SystemExit(
            "not a valid eval mix: the requested mask weights leave no eligible "
            "file (every file's modes resolved to 0). Add random_rect/whole_file "
            "weight, or supply provenance pools."
        )
    if skipped_files:
        print(
            f"[coverage] evaluating {len(eligible_files)}/{len(TRAINABLE_EXPORT_SPECS)} "
            f"files under this mask mix; skipping (no eligible mode): "
            f"{', '.join(skipped_files)}",
            flush=True,
        )
    sampler = SameFileBatchSampler(
        dataset.items, batch_size=args.batch, shuffle=False,
        include_files=set(eligible_files),
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
