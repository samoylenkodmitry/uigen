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
from atlas_ai.v7_batching import SameFileBatchSampler, WeightedSameFileBatchSampler
from atlas_ai.v7_masks import V7MaskWeights
from models.losses_v7 import support_masked_l1_loss, support_masked_sobel_mae
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
    out = {"l1": l1.detach()}
    if sobel_weight > 0.0:
        sobel = support_masked_sobel_mae(final_rgb, target_rgb, support)
        total = l1 + sobel_weight * sobel
        out["sobel"] = sobel.detach()
    else:
        total = l1
        out["sobel"] = torch.zeros((), device=device)
    out["total"] = total
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
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--snapshot-every", type=int, default=0)
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
                        help="Edge-precision loss weight. When >0, training total = "
                             "l1 + sobel_weight * support_masked_sobel_mae. Default 0 "
                             "keeps the historical L1-only recipe.")
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
    args = parser.parse_args()

    device = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

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
        file_weights = dict(DEFAULT_FILE_WEIGHTS)
        if args.file_sampling_weights:
            import yaml
            override = yaml.safe_load(Path(args.file_sampling_weights).read_text(encoding="utf-8"))
            if not isinstance(override, dict):
                raise SystemExit(f"{args.file_sampling_weights}: expected a mapping")
            file_weights.update({str(k): float(v) for k, v in override.items()})
        sampler = WeightedSameFileBatchSampler(
            dataset.items, batch_size=args.batch,
            file_weights=file_weights, num_batches=args.steps,
            generator=generator,
        )
        print(f"weighted sampling: {sampler.num_batches} batches, "
              f"file weights {dict(zip(sampler.files, [round(float(p), 3) for p in sampler.probs.tolist()]))}")
    else:
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

    if args.resume_from:
        from safetensors.torch import load_file
        resume_state = load_file(str(Path(args.resume_from)))
        version = int(resume_state.get("model_version", torch.tensor([0])).reshape(-1)[0].item())
        if version != 70:
            raise SystemExit(f"--resume-from expects V7 completer (version 70), got {version}")
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
    weighted_mode = args.sampling_mode == "weighted"
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        while step < args.steps:
            dataset.set_epoch(epoch)
            for batch in loader:
                if step >= args.steps:
                    break
                # Weighted sampling samples each file with replacement: the
                # same (skin, file) index can recur many times in one run.
                # Vary dataset.epoch per step so masks stay fresh.
                if weighted_mode:
                    dataset.set_epoch(step)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                    losses = compute_step_loss(
                        model, batch, support_masks, device,
                        sobel_weight=args.sobel_weight,
                    )
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
