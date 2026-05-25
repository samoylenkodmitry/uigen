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
from torch.utils.data._utils.collate import default_collate

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from atlas_ai.dataset_v7_completion import V7CompletionDataset
from atlas_ai.export_spec import TRAINABLE_EXPORT_SPECS
from atlas_ai.support_mask import load_support_masks
from atlas_ai.v7_batching import SameFileBatchSampler, WeightedSameFileBatchSampler
from atlas_ai.v7_masks import V7MaskWeights, has_alternatives, has_available_mask_mode
from models.losses_v7 import (
    hidden_supported_l1_loss,
    hidden_supported_l1_per_item,
    hidden_supported_sobel_mae,
    hidden_supported_sobel_mae_per_item,
    observed_passthrough_mae,
    support_masked_l1_loss,
    support_masked_l1_per_item,
    support_masked_sobel_mae,
    support_masked_sobel_mae_per_item,
)
from models.v7_completer import V7Completer


# Default file sampling weights for the weighted sampler. Hard / large files
# get more updates than tiny / simple ones. Matches the V7 handoff
# recommendation that was derived from the EQMAIN-only probe diagnosis.
DEFAULT_FILE_WEIGHTS: dict[str, float] = {
    "EQMAIN.bmp":   8.0,
    "MAIN.bmp":     4.0,
    "TITLEBAR.bmp": 4.0,
    "CBUTTONS.bmp": 4.0,
    "SHUFREP.bmp":  4.0,
    "PLEDIT.bmp":   4.0,
    "VOLUME.bmp":   3.0,
    "BALANCE.bmp":  3.0,
    "MONOSTER.bmp": 2.0,
    "POSBAR.bmp":   1.0,
    "PLAYPAUS.bmp": 1.0,
}


FILE_TO_ID: dict[str, int] = {spec.file_name: idx for idx, spec in enumerate(TRAINABLE_EXPORT_SPECS)}


def build_file_weights(
    yaml_path: str | Path | None,
    mode: str,
    *,
    defaults: dict[str, float] = DEFAULT_FILE_WEIGHTS,
) -> dict[str, float]:
    """Resolve the trainer's effective `file_weights` dict.

    mode='merge'   -> start from `defaults`, overlay YAML keys.
    mode='replace' -> start empty; ONLY YAML keys are sampled.

    Returns a dict[file_name -> non-negative weight]. The sampler then
    drops files with weight <= 0 and normalizes the rest.
    """
    if mode not in ("merge", "replace"):
        raise ValueError(f"mode must be 'merge' or 'replace', got {mode!r}")
    if mode == "replace" and yaml_path is None:
        raise ValueError(
            "replace mode requires a YAML path; otherwise every file gets weight 0"
        )
    if mode == "merge":
        weights = dict(defaults)
    else:
        weights = {}
    if yaml_path is not None:
        import yaml
        override = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
        if not isinstance(override, dict):
            raise ValueError(f"{yaml_path}: expected a mapping, got {type(override).__name__}")
        weights.update({str(k): float(v) for k, v in override.items()})
    return weights


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
    *,
    sobel_weight: float = 0.0,
    full_supported_weight: float = 0.05,
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
    skin_id_tensor: torch.Tensor | None = None
    if model.num_skins > 0:
        skin_index = batch.get("skin_index")
        if skin_index is None:
            raise RuntimeError(
                "model has skin conditioning but batch lacks 'skin_index'"
            )
        if not isinstance(skin_index, torch.Tensor):
            skin_index = torch.as_tensor(skin_index, dtype=torch.long)
        skin_id_tensor = skin_index.to(device=device, dtype=torch.long)
    final_rgb = model(observed_rgb, observed_mask, file_id, skin_id=skin_id_tensor)
    support = support_masks[file_name].to(device=device, dtype=final_rgb.dtype)
    # Primary loss is hidden-normalized: it scores only the pixels the model
    # actually generates (hidden = (1 - observed) * support). The full-supported
    # L1 is kept as a small anchor and as a debug metric — on its own it
    # dilutes mostly-observed samples because the observed pixels are hard
    # copies of the target.
    hidden_l1 = hidden_supported_l1_loss(final_rgb, target_rgb, observed_mask, support)
    hidden_l1_per_item, _has_hidden = hidden_supported_l1_per_item(
        final_rgb, target_rgb, observed_mask, support,
    )
    full_l1 = support_masked_l1_loss(final_rgb, target_rgb, support)
    full_l1_per_item = support_masked_l1_per_item(final_rgb, target_rgb, support)
    total = hidden_l1 + full_supported_weight * full_l1
    total_per_item = hidden_l1_per_item + full_supported_weight * full_l1_per_item
    out = {
        "l1": hidden_l1.detach(),          # primary (hidden-normalized) L1
        "hidden_l1": hidden_l1.detach(),
        "full_l1": full_l1.detach(),       # debug / secondary
    }
    if sobel_weight > 0.0:
        hidden_sobel = hidden_supported_sobel_mae(final_rgb, target_rgb, observed_mask, support)
        hidden_sobel_per_item, _ = hidden_supported_sobel_mae_per_item(
            final_rgb, target_rgb, observed_mask, support,
        )
        total = total + sobel_weight * hidden_sobel
        total_per_item = total_per_item + sobel_weight * hidden_sobel_per_item
        out["sobel"] = hidden_sobel.detach()
    else:
        out["sobel"] = torch.zeros((), device=device)
    # Diagnostic: with the hard copy intact this is ~0; a non-zero value means
    # the observed-pixel passthrough is broken.
    out["obs_passthrough"] = observed_passthrough_mae(
        final_rgb, target_rgb, observed_mask, support,
    ).detach()
    out["total"] = total
    out["total_per_item"] = total_per_item.detach()
    return out


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
    parser.add_argument("--skin-embedding-dim", type=int, default=0,
                        help="Width of the per-skin embedding. 0 disables skin "
                             "conditioning (Gate A behavior). >0 builds the "
                             "model with num_skins=len(skin_sources) and "
                             "passes skin_index from the batch in forward(). "
                             "Required for multi-skin Gate B training.")
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--snapshot-every", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=None,
                        help="Alias for --snapshot-every (overrides it when set). "
                             "Cloud-runner naming convention.")
    parser.add_argument("--stop-after-minutes", type=float, default=0.0,
                        help="Exit cleanly after this many minutes, saving "
                             "last.safetensors. 0 = no limit.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # V7 mask mix weights (defaults match the V7 plan).
    parser.add_argument("--mask-provenance", type=float, default=0.40)
    parser.add_argument("--mask-state-family", type=float, default=0.35)
    parser.add_argument("--mask-random-rect", type=float, default=0.20)
    parser.add_argument("--mask-whole-file", type=float, default=0.05)
    parser.add_argument("--mask-passthrough", type=float, default=0.0,
                        help="Diagnostic mode: mask=ones inside the support; >0 enables "
                             "the all-observed copy-through probe.")
    parser.add_argument("--only-file", default=None,
                        help="Filter the dataset to one file_name (e.g. EQMAIN.bmp) for "
                             "per-file overfit probes.")
    parser.add_argument("--resume-from", default=None,
                        help="Path to a V7 completer .safetensors checkpoint whose weights "
                             "seed the model. Steps restart at 0; the new --out is "
                             "independent of the source run.")
    parser.add_argument("--sobel-weight", type=float, default=0.0,
                        help="Edge-precision loss weight. When >0, training total adds "
                             "sobel_weight * hidden_supported_sobel_mae. Default 0 keeps "
                             "the L1-only recipe.")
    parser.add_argument("--full-supported-weight", type=float, default=0.05,
                        help="Weight on the full-supported L1 anchor term. The primary "
                             "loss is hidden-normalized; this small term keeps the "
                             "observed-pixel copy honest. Set 0 to drop it.")
    parser.add_argument("--pixel-hit-weight", type=float, default=0.0,
                        help="Reserved for a future pixel-margin loss attacking pixels "
                             "just over the 5/255 hit5 threshold. Not yet implemented.")
    parser.add_argument("--sampling-mode", choices=["epoch", "weighted"], default="epoch",
                        help="Batch sampling strategy. 'epoch' = round-robin "
                             "(SameFileBatchSampler), 'weighted' = file-weighted "
                             "with replacement (WeightedSameFileBatchSampler).")
    parser.add_argument("--file-sampling-weights", default=None,
                        help="Path to a YAML file mapping file_name -> weight. "
                             "Overrides the built-in default. Only used in weighted mode.")
    parser.add_argument("--file-sampling-weights-mode",
                        choices=["merge", "replace"], default="merge",
                        help="How --file-sampling-weights interacts with the built-in "
                             "DEFAULT_FILE_WEIGHTS. 'merge' (default, backward-compatible): "
                             "YAML keys override defaults; files absent from YAML keep their "
                             "default weight and stay in the sampling pool. 'replace': files "
                             "absent from YAML get weight 0 and are dropped from the sampler. "
                             "Use 'replace' for focused probes that need to train on a strict "
                             "subset of files only.")
    parser.add_argument("--within-file-replacement", dest="within_file_replacement",
                        type=lambda s: str(s).lower() not in ("0", "false", "no"),
                        default=True,
                        help="Weighted-mode only. When True (default) a same-file batch samples "
                             "indices with replacement (required for groups smaller than batch). "
                             "Pass --within-file-replacement false to draw distinct indices when "
                             "the group is at least batch_size — useful for multi-skin Gate B "
                             "where one batch should cover distinct skins instead of duplicates.")
    parser.add_argument("--progress-every", type=int, default=200,
                        help="Print a step/loss/ETA progress line to stdout every N steps. "
                             "0 disables. Use small values for short cloud runs so the kernel "
                             "log isn't silent.")
    args = parser.parse_args()

    device = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # --save-every is an alias for --snapshot-every (cloud-runner naming).
    if args.save_every is not None:
        args.snapshot_every = args.save_every

    skin_sources = parse_skin_sources(args.skin_sources)
    mask_weights = V7MaskWeights(
        provenance=args.mask_provenance,
        state_family=args.mask_state_family,
        random_rect=args.mask_random_rect,
        whole_file=args.mask_whole_file,
        passthrough=args.mask_passthrough,
    )
    dataset = V7CompletionDataset(
        skin_sources=skin_sources,
        state_families_path=args.state_families,
        mask_weights=mask_weights,
        seed=args.seed,
    )
    if args.only_file:
        keep = args.only_file
        dataset.items = [(s, f) for (s, f) in dataset.items if f == keep]
        if not dataset.items:
            raise SystemExit(f"--only-file {keep!r}: no matching items in dataset")
        print(f"--only-file {keep!r}: kept {len(dataset.items)} item(s)")
    generator = torch.Generator().manual_seed(args.seed)
    if args.sampling_mode == "weighted":
        weights_mode = args.file_sampling_weights_mode
        try:
            file_weights = build_file_weights(
                args.file_sampling_weights, weights_mode,
            )
        except ValueError as e:
            raise SystemExit(str(e))
        sampler = WeightedSameFileBatchSampler(
            dataset.items, batch_size=args.batch,
            file_weights=file_weights, num_batches=args.steps,
            generator=generator,
            within_file_replacement=args.within_file_replacement,
        )
        # Log the effective normalized probability per file so the run log
        # makes the actual sampling distribution visible — catches mistakes
        # like "I wrote a strip-only YAML but merge mode kept other files".
        eff_probs = sorted(
            zip(sampler.files, [float(p) for p in sampler.probs.tolist()]),
            key=lambda kv: -kv[1],
        )
        print(f"weighted sampling: mode={weights_mode}  batches={sampler.num_batches}  "
              f"within_file_replacement={sampler.within_file_replacement}")
        print("effective file probabilities (post-normalization):")
        for fn, p in eff_probs:
            print(f"  {fn:20s} {p:7.4f}")
    else:
        sampler = SameFileBatchSampler(
            dataset.items, batch_size=args.batch, shuffle=True, generator=generator,
        )
    # Guard: every file that will actually be sampled must have at least one
    # eligible mask mode under the configured weights. Otherwise __getitem__
    # raises mid-run — e.g. a state_family-only mix sampling a component-only
    # file (POSBAR/MAIN/TITLEBAR/PLEDIT have no `alternatives` family). Fail
    # fast at startup instead of crashing thousands of steps in.
    if args.sampling_mode == "weighted":
        sampled_files = list(sampler.files)
    else:
        sampled_files = sorted({f for (_s, f) in dataset.items})
    ineligible = [
        f for f in sampled_files
        if not has_available_mask_mode(
            mask_weights,
            have_provenance=bool(dataset.provenance_pools.get(f)),
            have_state_family=has_alternatives(dataset.state_families.get(f, [])),
        )
    ]
    if ineligible:
        raise SystemExit(
            "invalid mask mix: no eligible mask mode for sampled file(s) "
            f"{', '.join(sorted(ineligible))}. They have no `alternatives` "
            "family, so a state_family-only mix leaves nothing to sample. Add "
            "--mask-random-rect > 0 (and/or --mask-whole-file/--mask-provenance), "
            "or drop these files from --file-sampling-weights."
        )
    if args.sampling_mode == "weighted":
        # WeightedSameFileBatchSampler intentionally reuses the same
        # (skin,file) index many times. Build each batch after setting the
        # per-step mask epoch so __getitem__ sees the fresh RNG stream.
        # A normal DataLoader would fetch the batch before the training loop
        # can call dataset.set_epoch(step), and worker copies would not see
        # subsequent set_epoch() calls at all.
        loader = None
        if args.num_workers:
            print(
                "weighted sampling uses direct indexed collation so masks "
                f"can change every step; ignoring --num-workers={args.num_workers}",
                flush=True,
            )
    else:
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory and device.type == "cuda",
        )
    support_masks = load_support_masks()

    num_skins = len(dataset.skin_ids) if args.skin_embedding_dim > 0 else 0
    model = V7Completer(
        base_channels=args.base_channels,
        file_embedding_dim=args.file_embedding_dim,
        num_skins=num_skins,
        skin_embedding_dim=args.skin_embedding_dim,
    ).to(device)

    if args.resume_from:
        from safetensors.torch import load_file
        resume_state = load_file(str(Path(args.resume_from)))
        version = int(resume_state.get("model_version", torch.tensor([0])).reshape(-1)[0].item())
        if version not in (70, 71):
            raise SystemExit(f"--resume-from expects V7 completer (version 70 or 71), got {version}")
        # Validate skin conditioning shape matches between the source and
        # destination model, otherwise the embedding table sizes disagree
        # and load_state_dict would either crash or silently truncate.
        src_num_skins = int(
            resume_state.get("num_skins_buffer", torch.tensor([0])).reshape(-1)[0].item()
        )
        src_skin_dim = int(
            resume_state.get("skin_embedding_dim_buffer", torch.tensor([0])).reshape(-1)[0].item()
        )
        if src_num_skins != num_skins or src_skin_dim != model.skin_embedding_dim:
            raise SystemExit(
                f"--resume-from checkpoint has num_skins={src_num_skins} skin_dim={src_skin_dim}, "
                f"but this run expects num_skins={num_skins} skin_dim={model.skin_embedding_dim}"
            )
        missing, unexpected = model.load_state_dict(resume_state, strict=False)
        if missing:
            raise SystemExit(f"--resume-from missing keys: {sorted(missing)[:5]}...")
        if unexpected:
            raise SystemExit(f"--resume-from unexpected keys: {sorted(unexpected)[:5]}...")
        print(f"resumed weights from {args.resume_from}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    env_info: dict = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            env_info["cuda_device_name"] = torch.cuda.get_device_name(device)
            props = torch.cuda.get_device_properties(device)
            env_info["cuda_total_memory_mib"] = props.total_memory // (1024 * 1024)
        except Exception:
            pass
    try:
        import platform as _platform
        env_info["python_version"] = _platform.python_version()
        env_info["platform"] = _platform.platform()
    except Exception:
        pass

    config = {
        "model_version": 71 if num_skins > 0 else 70,
        "num_skins": num_skins,
        "skin_embedding_dim": model.skin_embedding_dim,
        "skin_id_to_index": dict(dataset.skin_id_to_index),
        "args": vars(args),
        "skin_sources": skin_sources,
        "dataset_size": len(dataset),
        "batches_per_epoch": len(sampler),
        "env": env_info,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))

    import time as _time
    metrics_path = out / "metrics.jsonl"
    best = float("inf")
    step = 0
    epoch = 0
    model.train()
    weighted_mode = args.sampling_mode == "weighted"
    start_time = _time.monotonic()
    stop_after_seconds = float(args.stop_after_minutes) * 60.0
    stopped_by_time = False
    progress_every = int(getattr(args, "progress_every", 200) or 200)
    recent_losses: list[float] = []
    recent_window = max(progress_every, 1)
    last_progress_time = start_time
    # Running per-(mode|file|skin) sums and counts. Reset after each progress
    # line so each emitted block describes the *window* since the previous
    # print. Single skin runs may have one entry; that's fine.
    skin_index_to_id = {idx: sid for sid, idx in dataset.skin_id_to_index.items()}

    def _new_bucket() -> dict[str, list[float]]:
        return {"sum": [0.0], "n": [0]}

    mode_buckets: dict[str, dict[str, list[float]]] = {}
    file_buckets: dict[str, dict[str, list[float]]] = {}
    skin_buckets: dict[str, dict[str, list[float]]] = {}

    def _accumulate(b: dict, key: str, loss: float) -> None:
        slot = b.setdefault(key, _new_bucket())
        slot["sum"][0] += loss
        slot["n"][0] += 1

    def _drain(b: dict) -> dict[str, tuple[float, int]]:
        out_means: dict[str, tuple[float, int]] = {}
        for k, slot in b.items():
            n = slot["n"][0]
            if n > 0:
                out_means[k] = (slot["sum"][0] / n, n)
        b.clear()
        return out_means

    def _fmt_breakdown(label: str, means: dict[str, tuple[float, int]]) -> str:
        if not means:
            return f"  {label}: (no data)"
        items = sorted(means.items(), key=lambda kv: -kv[1][0])
        parts = [f"{k}={v:.4f}(n{n})" for k, (v, n) in items]
        return f"  {label}: " + " ".join(parts)

    print(f"training start: target steps={args.steps} batch={args.batch} "
          f"progress_every={progress_every}", flush=True)
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        while step < args.steps:
            if weighted_mode:
                batch_iter = iter(sampler)
            else:
                dataset.set_epoch(epoch)
                assert loader is not None
                batch_iter = iter(loader)
            for batch_or_indices in batch_iter:
                if step >= args.steps:
                    break
                if stop_after_seconds > 0 and (_time.monotonic() - start_time) >= stop_after_seconds:
                    stopped_by_time = True
                    break
                if weighted_mode:
                    # Weighted sampling samples each file with replacement:
                    # the same (skin, file) index can recur many times in one
                    # run. Vary dataset.epoch before __getitem__ so masks stay
                    # fresh for the current batch.
                    dataset.set_epoch(step)
                    batch = default_collate([
                        dataset[int(idx)] for idx in batch_or_indices
                    ])
                else:
                    batch = batch_or_indices
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                    losses = compute_step_loss(
                        model, batch, support_masks, device,
                        sobel_weight=args.sobel_weight,
                        full_supported_weight=args.full_supported_weight,
                    )
                if args.amp and device.type == "cuda":
                    scaler.scale(losses["total"]).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    losses["total"].backward()
                    optimizer.step()
                per_item_total = losses.pop("total_per_item").to("cpu")
                logged = {k: float(v.detach().cpu()) for k, v in losses.items()}
                logged["step"] = step
                logged["epoch"] = epoch
                logged["file_name"] = batch["file_name"][0] if isinstance(batch["file_name"], list) else batch["file_name"]
                # Bucket per-item losses by mode / file / skin. Same-file
                # batches mean every item shares file_name (already in
                # logged), but mode and skin vary per item.
                modes_in_batch = batch.get("mode", [])
                if not isinstance(modes_in_batch, (list, tuple)):
                    modes_in_batch = [modes_in_batch]
                skin_idx_batch = batch.get("skin_index")
                if isinstance(skin_idx_batch, torch.Tensor):
                    skin_idx_list = skin_idx_batch.detach().cpu().tolist()
                else:
                    skin_idx_list = list(skin_idx_batch) if skin_idx_batch is not None else []
                bs = per_item_total.shape[0]
                for i in range(bs):
                    loss_i = float(per_item_total[i].item())
                    mode_i = str(modes_in_batch[i]) if i < len(modes_in_batch) else "?"
                    _accumulate(mode_buckets, mode_i, loss_i)
                    _accumulate(file_buckets, logged["file_name"], loss_i)
                    if i < len(skin_idx_list):
                        sid = skin_index_to_id.get(int(skin_idx_list[i]), f"idx{int(skin_idx_list[i])}")
                        _accumulate(skin_buckets, sid, loss_i)
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
                recent_losses.append(logged["total"])
                if len(recent_losses) > recent_window:
                    recent_losses = recent_losses[-recent_window:]
                step += 1
                if progress_every > 0 and step % progress_every == 0:
                    now = _time.monotonic()
                    dt = now - last_progress_time
                    sec_per_step = dt / progress_every if progress_every > 0 else 0.0
                    remaining = max(args.steps - step, 0)
                    eta_sec = sec_per_step * remaining
                    eta_min = eta_sec / 60.0
                    mean_recent = sum(recent_losses) / max(len(recent_losses), 1)
                    pct = 100.0 * step / max(args.steps, 1)
                    elapsed_min = (now - start_time) / 60.0
                    mode_means = _drain(mode_buckets)
                    file_means = _drain(file_buckets)
                    skin_means = _drain(skin_buckets)
                    print(
                        f"[step {step:>6d}/{args.steps}  {pct:5.1f}%]  "
                        f"loss(mean{len(recent_losses):>4d})={mean_recent:.4f}  "
                        f"best={best:.4f}  "
                        f"sec/step={sec_per_step:.3f}  "
                        f"elapsed={elapsed_min:6.1f}min  ETA={eta_min:6.1f}min",
                        flush=True,
                    )
                    print(_fmt_breakdown("by_mode", mode_means), flush=True)
                    print(_fmt_breakdown("by_file", file_means), flush=True)
                    print(_fmt_breakdown("by_skin", skin_means), flush=True)
                    last_progress_time = now
            if stopped_by_time:
                break
            epoch += 1
    save_state_dict(out / "last.safetensors", model.state_dict())
    reason = "stopped by --stop-after-minutes" if stopped_by_time else "finished"
    print(f"trained V7Completer for {step} step(s) ({reason}); best step loss {best:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
