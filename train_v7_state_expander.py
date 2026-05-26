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
import torch.nn.functional as F
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


def compute_loss(model, batch, support_masks, device, *, sobel_weight: float,
                 gate_loss_weight: float = 0.0, gate_change_threshold: float = 0.02):
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
    gate = gate_logits = None
    if model.output_mode == "gated":
        pred, gate, gate_logits = model(source, source_idx, target_idx, family_id, file_id,
                                        skin_id=skin_id, return_gate=True)
    else:
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
    if gate is not None:
        with torch.no_grad():
            changed = ((source - target).abs().amax(dim=1, keepdim=True) > 0.02) & (support > 0.5)
            unchanged = (~((source - target).abs().amax(dim=1, keepdim=True) > 0.02)) & (support > 0.5)
            out["gate_mean"] = gate.mean().detach()
            out["gate_p90"] = torch.quantile(gate.flatten(), 0.9).detach()
            out["gate_changed"] = (gate[changed].mean() if changed.any()
                                   else torch.zeros((), device=device)).detach()
            out["gate_unchanged"] = (gate[unchanged].mean() if unchanged.any()
                                     else torch.zeros((), device=device)).detach()
    # Optional gate supervision: BCE pushing the gate open on changed pixels.
    if gate_logits is not None and gate_loss_weight > 0.0:
        sup_b = support > 0.5
        with torch.no_grad():
            changed_target = ((source - target).abs().amax(dim=1, keepdim=True)
                              > gate_change_threshold) & sup_b
            n_pos = changed_target.sum().clamp_min(1.0)
            n_neg = (sup_b.sum() - changed_target.sum()).clamp_min(0.0)
            pos_weight = torch.clamp(n_neg / n_pos, max=10.0)
        gate_l = F.binary_cross_entropy_with_logits(
            gate_logits[sup_b], changed_target[sup_b].float(), pos_weight=pos_weight)
        total = total + gate_loss_weight * gate_l
        out["gate_loss"] = gate_l.detach()
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
    p.add_argument("--output-mode", choices=["residual", "direct", "unbounded", "gated"],
                   default="gated",
                   help="Output head (default gated, the validated state-expansion "
                        "head): gated=(1-g)*source+g*sigmoid(rgb) with a copy-biased "
                        "gate; residual=clamp(source+tanh(delta)); direct=sigmoid(logits); "
                        "unbounded=clamp(source+delta). residual/direct/unbounded kept "
                        "for ablation.")
    p.add_argument("--no-identity", action="store_true",
                   help="Exclude i==i pairs entirely (default includes them, "
                        "downweighted by --identity-weight).")
    p.add_argument("--identity-weight", type=float, default=0.1,
                   help="Within-family sampling weight for identity (i==i) pairs "
                        "relative to transition pairs (weight 1.0). Low by default "
                        "so S1 trains transitions; residual-from-source already "
                        "handles copying unchanged content. Ignored with --no-identity.")
    p.add_argument("--only-family", default=None,
                   help="Restrict training to one family_key (e.g. 'CBUTTONS/play') "
                        "for single-family overfit diagnostics.")
    p.add_argument("--gate-loss-weight", type=float, default=0.0,
                   help="Optional supervision for the gated head: BCE pushing the "
                        "gate open on changed pixels. 0 disables (let reconstruction "
                        "decide); 0.05 is a reasonable on value. Only used in gated mode.")
    p.add_argument("--gate-change-threshold", type=float, default=0.02,
                   help="|source-target| (max over channels) above which a supported "
                        "pixel is a gate-supervision target. Only used with "
                        "--gate-loss-weight > 0.")
    # --- S2 split / sampling controls (default off -> S1 behavior) ---
    p.add_argument("--heldout-skins", default=None,
                   help="Comma-separated skin ids held out entirely for the "
                        "deployability (unseen-style) eval split. Never trained.")
    p.add_argument("--heldout-skin-fraction", type=float, default=0.0,
                   help="If --heldout-skins not given, deterministically hold out "
                        "this fraction of skins (by --seed).")
    p.add_argument("--seen-pair-val-fraction", type=float, default=0.0,
                   help="Fraction of each big family's (i,j) pairs held out across "
                        "train skins for the seen-skin/unseen-pair eval split.")
    p.add_argument("--family-weights", default=None,
                   help="YAML mapping family_key -> difficulty weight (sampler "
                        "key_weights). Missing families default to 1.0.")
    p.add_argument("--local-pair-prob", type=float, default=0.5,
                   help="Within a family, probability mass on local (|i-j|<=delta) "
                        "transitions vs global. Balances neighbour vs jump moves.")
    p.add_argument("--local-delta", type=int, default=2,
                   help="|i-j| <= local-delta counts as a local transition.")
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
    # Resolve held-out skins (explicit list wins; else a deterministic fraction).
    all_skin_ids = sorted(skin_sources.keys())
    if args.heldout_skins:
        heldout = [s.strip() for s in args.heldout_skins.split(",") if s.strip()]
        unknown = [s for s in heldout if s not in skin_sources]
        if unknown:
            raise SystemExit(f"--heldout-skins not in skin set: {unknown}")
    elif args.heldout_skin_fraction > 0:
        import random as _r
        k = max(1, int(round(args.heldout_skin_fraction * len(all_skin_ids))))
        heldout = sorted(_r.Random(args.seed).sample(all_skin_ids, k))
    else:
        heldout = []
    ds = StatePairsDataset(
        skin_sources=skin_sources,
        state_families_path=args.state_families,
        include_identity=not args.no_identity,
        heldout_skins=heldout,
        heldout_pair_fraction=args.seen_pair_val_fraction,
        local_delta=args.local_delta,
        split_seed=args.seed,
    )
    num_skins = len(ds.skin_ids) if args.skin_embedding_dim > 0 else 0
    s2_mode = bool(heldout or args.seen_pair_val_fraction > 0 or args.family_weights)
    from collections import Counter as _Counter
    split_counts = _Counter(ds.split_of)
    print(f"splits: {dict(split_counts)}; heldout_skins={heldout or '(none)'}", flush=True)
    model = V7StateExpander(
        num_families=ds.num_families,
        max_frames=ds.max_frames,
        base_channels=args.base_channels,
        file_embedding_dim=args.file_embedding_dim,
        family_embedding_dim=args.family_embedding_dim,
        frame_embedding_dim=args.frame_embedding_dim,
        num_skins=num_skins,
        skin_embedding_dim=args.skin_embedding_dim,
        output_mode=args.output_mode,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    support_masks = load_support_masks()

    generator = torch.Generator().manual_seed(args.seed)
    all_keys = {f.key for f in ds.alt_families}
    if s2_mode:
        # S2: train only on the 'train' split (item_weights zero elsewhere),
        # balance local/global transitions within each family, and apply
        # per-family difficulty weights as the sampler's key_weights.
        item_weights = ds.training_item_weights(local_pair_prob=args.local_pair_prob)
        family_weights = {}
        if args.family_weights:
            import yaml as _yaml
            family_weights = {str(k): float(v) for k, v in
                              (_yaml.safe_load(Path(args.family_weights).read_text()) or {}).items()}
        key_weights = {k: family_weights.get(k, 1.0) for k in all_keys}
        if args.only_family:
            key_weights = {k: (key_weights[k] if k == args.only_family else 0.0) for k in all_keys}
        print(f"S2 sampling: train-split only, local_pair_prob={args.local_pair_prob}, "
              f"family difficulty weights from "
              f"{args.family_weights or '(uniform)'}", flush=True)
        nonuniform = {k: family_weights[k] for k in sorted(family_weights) if family_weights[k] != 1.0}
        if nonuniform:
            print(f"  family_weights != 1.0: {nonuniform}", flush=True)
    else:
        # S1 behavior: family-balanced, identity downweighted.
        item_weights = [
            (args.identity_weight if i == j else 1.0) for (_sid, _fid, i, j) in ds.items
        ]
        key_weights = None
        if args.only_family:
            if args.only_family not in all_keys:
                raise SystemExit(f"--only-family {args.only_family!r} not in {sorted(all_keys)}")
            key_weights = {k: (1.0 if k == args.only_family else 0.0) for k in all_keys}
            print(f"--only-family: restricting to {args.only_family}", flush=True)
    sampler = WeightedSameKeyBatchSampler(
        ds.group_keys, batch_size=args.batch, num_batches=args.steps,
        key_weights=key_weights, item_weights=item_weights, generator=generator,
    )
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=args.num_workers,
                        collate_fn=default_collate)
    print(f"family-balanced sampling: {len(sampler.keys)} families", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = {
        "model": "V7StateExpander", "model_version": 72,
        "num_families": ds.num_families, "max_frames": ds.max_frames,
        "num_skins": num_skins, "skin_id_to_index": ds.skin_id_to_index,
        "alt_families": [(f.file_name, f.family, f.num_frames) for f in ds.alt_families],
        "n_items": len(ds), "args": vars(args),
        "heldout_skins": heldout, "split_counts": dict(split_counts),
        "s2_mode": s2_mode,
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
        losses = compute_loss(model, batch, support_masks, device,
                              sobel_weight=args.sobel_weight,
                              gate_loss_weight=args.gate_loss_weight,
                              gate_change_threshold=args.gate_change_threshold)
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
            if "gate_mean" in logged:
                print(f"  gate: mean={logged['gate_mean']:.3f} p90={logged['gate_p90']:.3f} "
                      f"on_changed={logged['gate_changed']:.3f} "
                      f"on_unchanged={logged['gate_unchanged']:.3f}", flush=True)
            last_t = now
    save_state_dict(out / "last.safetensors", model.state_dict())
    metrics_file.close()
    print(f"trained V7StateExpander for {step} step(s); best step loss {best:.6f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
