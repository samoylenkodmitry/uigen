#!/usr/bin/env python3
"""Train the V7.1 StateFamilyExpander on alternatives-family frame pairs.

Task: (source frame, source_idx, target_idx, family_id, file_id[, skin_id]) ->
target frame. Loss is support-masked L1 (+ optional Sobel) over the target
rect's supported pixels. There is no observed/hidden split here — the whole
target frame is predicted — so the plain support-masked metric is the right one.

Oracle note: --skin-embedding-dim > 0 conditions on an ORACLE skin id. Fine for
the Gate S1/S2 capacity test, not deployable.

Emits live stdout progress (per project convention): a start line, a periodic
line every --progress-every steps with running-mean loss / sec-per-step /
ETA / per-family breakdown, and an end line.
"""

from __future__ import annotations

import argparse
import json
import sys
import time as _time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from atlas_ai.state_pairs_dataset import StatePairsDataset
from atlas_ai.support_mask import load_support_masks
from atlas_ai.v7_batching import WeightedSameKeyBatchSampler
from models.losses_v7 import (
    support_masked_l1_loss,
    support_masked_l1_per_item,
    support_masked_sobel_mae,
    support_masked_sobel_mae_per_item,
)
from models.v7_state_expander import V7StateExpander


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


def save_state_dict(path: Path, state: dict) -> None:
    from safetensors.torch import save_file
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.cpu().contiguous() for k, v in state.items()}, str(path))


def compute_loss(model, batch, support_masks, device, *, sobel_weight: float):
    source = batch["source_rgb"].to(device, non_blocking=True)
    target = batch["target_rgb"].to(device, non_blocking=True)
    support = batch["target_support"].to(device, non_blocking=True)  # [B,1,H,W]
    source_idx = batch["source_idx"].to(device)
    target_idx = batch["target_idx"].to(device)
    family_id = batch["family_id"].to(device)
    file_id = batch["file_id"].to(device)
    skin_id = None
    if model.num_skins > 0:
        skin_id = batch["skin_index"].to(device=device, dtype=torch.long)
    pred = model(source, source_idx, target_idx, family_id, file_id, skin_id=skin_id)
    l1 = support_masked_l1_loss(pred, target, support)
    l1_pi = support_masked_l1_per_item(pred, target, support)
    out = {"l1": l1.detach()}
    if sobel_weight > 0.0:
        sob = support_masked_sobel_mae(pred, target, support)
        sob_pi = support_masked_sobel_mae_per_item(pred, target, support)
        total = l1 + sobel_weight * sob
        total_pi = l1_pi + sobel_weight * sob_pi
        out["sobel"] = sob.detach()
    else:
        total = l1
        total_pi = l1_pi
        out["sobel"] = torch.zeros((), device=device)
    out["total"] = total
    out["total_per_item"] = total_pi.detach()
    return out


def _accumulate(buckets: dict, key: str, val: float) -> None:
    s = buckets.setdefault(key, [0.0, 0])
    s[0] += val
    s[1] += 1


def _drain(buckets: dict) -> dict[str, float]:
    means = {k: (s[0] / s[1] if s[1] else float("nan")) for k, s in buckets.items()}
    buckets.clear()
    return means


def _fmt(label: str, means: dict[str, float]) -> str:
    items = sorted(means.items(), key=lambda kv: -kv[1])
    return f"  {label}: " + " ".join(f"{k}={v:.4f}" for k, v in items)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state-families", default="configs/state_families_classic.yaml")
    p.add_argument("--skin-sources", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--base-channels", type=int, default=48)
    p.add_argument("--file-embedding-dim", type=int, default=16)
    p.add_argument("--family-embedding-dim", type=int, default=16)
    p.add_argument("--frame-embedding-dim", type=int, default=16)
    p.add_argument("--skin-embedding-dim", type=int, default=0,
                   help="ORACLE skin embedding width. 0 disables skin conditioning.")
    p.add_argument("--sobel-weight", type=float, default=0.25)
    p.add_argument("--no-identity", action="store_true",
                   help="Exclude i==i pairs entirely (default includes them, "
                        "downweighted by --identity-weight).")
    p.add_argument("--identity-weight", type=float, default=0.1,
                   help="Within-family sampling weight for identity (i==i) pairs "
                        "relative to transition pairs (weight 1.0). Low by default "
                        "so S1 trains transitions; residual-from-source already "
                        "handles copying unchanged content. Ignored with --no-identity.")
    p.add_argument("--checkpoint-every", type=int, default=2000)
    p.add_argument("--snapshot-every", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=200)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    skin_sources = parse_skin_sources(args.skin_sources)
    ds = StatePairsDataset(
        skin_sources=skin_sources,
        state_families_path=args.state_families,
        include_identity=not args.no_identity,
    )
    num_skins = len(ds.skin_ids) if args.skin_embedding_dim > 0 else 0
    model = V7StateExpander(
        num_families=ds.num_families,
        max_frames=ds.max_frames,
        base_channels=args.base_channels,
        file_embedding_dim=args.file_embedding_dim,
        family_embedding_dim=args.family_embedding_dim,
        frame_embedding_dim=args.frame_embedding_dim,
        num_skins=num_skins,
        skin_embedding_dim=args.skin_embedding_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    support_masks = load_support_masks()

    generator = torch.Generator().manual_seed(args.seed)
    # Family-balanced exposure: equal probability per (file,family) so eject's
    # 4 pairs and VOLUME's 784 pairs get equal gradient steps. Identity pairs
    # are downweighted within each family (transitions are the S1 task).
    item_weights = [
        (args.identity_weight if i == j else 1.0) for (_sid, _fid, i, j) in ds.items
    ]
    sampler = WeightedSameKeyBatchSampler(
        ds.group_keys, batch_size=args.batch, num_batches=args.steps,
        item_weights=item_weights, generator=generator,
    )
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=args.num_workers,
                        collate_fn=default_collate)
    print(f"family-balanced sampling: {len(sampler.keys)} families, equal prob "
          f"({1.0/len(sampler.keys):.4f} each); identity_weight="
          f"{0.0 if args.no_identity else args.identity_weight}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = {
        "model": "V7StateExpander", "model_version": 72,
        "num_families": ds.num_families, "max_frames": ds.max_frames,
        "num_skins": num_skins, "skin_id_to_index": ds.skin_id_to_index,
        "alt_families": [(f.file_name, f.family, f.num_frames) for f in ds.alt_families],
        "n_items": len(ds), "args": vars(args),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))

    print(f"training start: V7StateExpander steps={args.steps} batch={args.batch} "
          f"lr={args.lr} c={args.base_channels} families={ds.num_families} "
          f"skins={len(ds.skin_ids)} (skin_emb={num_skins>0}) items={len(ds)} "
          f"progress_every={args.progress_every}", flush=True)

    metrics_file = (out / "metrics.jsonl").open("w")
    fam_buckets: dict = {}
    fam_batch_counts: dict[str, int] = {}
    recent: list[float] = []
    best = float("inf")
    step = 0
    start = _time.monotonic()
    last_t = start
    model.train()
    # The weighted sampler yields exactly args.steps batches, so a single pass
    # over the loader is the whole run (no epoch loop).
    for batch in loader:
        if step >= args.steps:
            break
        optimizer.zero_grad(set_to_none=True)
        losses = compute_loss(model, batch, support_masks, device, sobel_weight=args.sobel_weight)
        losses["total"].backward()
        optimizer.step()
        per_item = losses.pop("total_per_item").cpu()
        fams = batch["family_key"] if isinstance(batch["family_key"], list) else [batch["family_key"]]
        batch_family = fams[0] if fams else "?"
        fam_batch_counts[batch_family] = fam_batch_counts.get(batch_family, 0) + 1
        for i in range(per_item.shape[0]):
            _accumulate(fam_buckets, fams[i] if i < len(fams) else "?", float(per_item[i]))
        logged = {k: float(v.detach().cpu()) for k, v in losses.items()}
        logged.update(step=step, family=batch_family, file_name=batch["file_name"][0])
        metrics_file.write(json.dumps(logged) + "\n")
        if logged["total"] < best:
            best = logged["total"]
            save_state_dict(out / "best.safetensors", model.state_dict())
        if (step + 1) % args.checkpoint_every == 0 or step == 0:
            save_state_dict(out / "last.safetensors", model.state_dict())
        if args.snapshot_every > 0 and (step + 1) % args.snapshot_every == 0:
            save_state_dict(out / f"snapshot_step{step + 1:06d}.safetensors", model.state_dict())
        recent.append(logged["total"])
        recent = recent[-args.progress_every:] if args.progress_every > 0 else recent
        step += 1
        if args.progress_every > 0 and step % args.progress_every == 0:
            now = _time.monotonic()
            sps = (now - last_t) / args.progress_every
            eta = sps * max(args.steps - step, 0) / 60.0
            print(f"[step {step:>6d}/{args.steps}  {100.0*step/args.steps:5.1f}%]  "
                  f"loss(mean{len(recent):>4d})={sum(recent)/len(recent):.4f}  best={best:.4f}  "
                  f"sec/step={sps:.3f}  elapsed={(now-start)/60.0:5.1f}min  ETA={eta:5.1f}min",
                  flush=True)
            print(_fmt("by_family", _drain(fam_buckets)), flush=True)
            counts = sorted(fam_batch_counts.items(), key=lambda kv: kv[1])
            print("  family_batches(min->max): "
                  + " ".join(f"{k}={v}" for k, v in counts), flush=True)
            last_t = now
    save_state_dict(out / "last.safetensors", model.state_dict())
    metrics_file.close()
    print(f"trained V7StateExpander for {step} step(s); best step loss {best:.6f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
