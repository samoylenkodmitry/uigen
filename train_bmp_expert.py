#!/usr/bin/env python3
"""Train one V10 BMPExpert (full render -> exact BMP).

Loss (per HANDOFF_V10):
    L = 1.0 * L1_RGB  +  1.5 * Sobel_RGB  +  0.5 * Laplacian_RGB

Metrics:
    MAE, hit_5_255, Sobel MAE.

Emits live progress (per project convention): start line + periodic step lines
with running-mean loss / sec/step / ETA / metrics (flush=True).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time as _time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from atlas_ai.dataset_v10_bmp import BMPExpertDataset
from atlas_ai.export_spec import TRAINABLE_EXPORT_SPECS
from models.bmp_expert_net import BMPExpertNet, BMPPatchDiscriminator


SPEC_BY_NAME = {s.file_name: s for s in TRAINABLE_EXPORT_SPECS}


def _sobel(x: torch.Tensor) -> torch.Tensor:
    kx = x.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
    ky = x.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)
    gx = F.conv2d(x, kx.repeat(x.shape[1], 1, 1, 1), padding=1, groups=x.shape[1])
    gy = F.conv2d(x, ky.repeat(x.shape[1], 1, 1, 1), padding=1, groups=x.shape[1])
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


def _laplacian(x: torch.Tensor) -> torch.Tensor:
    k = x.new_tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).view(1, 1, 3, 3)
    return F.conv2d(x, k.repeat(x.shape[1], 1, 1, 1), padding=1, groups=x.shape[1])


def _hit_5_255(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(((pred - target).abs() * 255.0 <= 5.0).all(dim=1).float().mean())


def compute_losses(pred: torch.Tensor, target: torch.Tensor) -> dict:
    l1 = F.l1_loss(pred, target)
    sob = (_sobel(pred) - _sobel(target)).abs().mean()
    lap = (_laplacian(pred) - _laplacian(target)).abs().mean()
    total = l1 + 1.5 * sob + 0.5 * lap
    with torch.no_grad():
        mae = (pred - target).abs().mean()
        hit5 = _hit_5_255(pred, target)
    return {"total": total, "l1": l1.detach(), "sobel": sob.detach(),
            "laplacian": lap.detach(), "mae": mae, "hit_5_255": torch.tensor(hit5)}


def save_state_dict(path: Path, state: dict) -> None:
    from safetensors.torch import save_file
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.cpu().contiguous() for k, v in state.items()}, str(path))


@torch.no_grad()
def evaluate_subset(model, eval_loader, device, autocast_ctx) -> tuple[float, float]:
    """Mean MAE and hit_5_255 over a fixed eval subset (no grad). Restores
    train mode on exit."""
    model.eval()
    tot_mae = tot_hit = 0.0
    n = 0
    for batch in eval_loader:
        x = batch["render"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)
        with autocast_ctx():
            p = model(x).float().clamp(0.0, 1.0)
        b = x.shape[0]
        tot_mae += float((p - y).abs().mean(dim=(1, 2, 3)).sum())
        tot_hit += float(((p - y).abs() * 255.0 <= 5.0).all(dim=1).float().mean(dim=(1, 2)).sum())
        n += b
    model.train()
    return (tot_mae / max(1, n), tot_hit / max(1, n))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, help="V10 dataset root (e.g., data_v10/).")
    p.add_argument("--bmp", required=True, help="Target BMP file name (e.g., MAIN.bmp).")
    p.add_argument("--out", required=True, help="Run output dir for checkpoints/metrics.")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--base", type=int, default=48)
    p.add_argument("--attn-dim", type=int, default=256)
    p.add_argument("--dec-ch", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--attn-layers", type=int, default=2)
    p.add_argument("--query-div", type=int, default=4,
                   help="Query grid = target H/Q x W/Q. 4 (default) for small BMPs.")
    p.add_argument("--kv-scale", type=int, default=1,
                   help="Multiply cross-attention K/V pool resolution. 1 = default "
                        "(~2.4k tokens, lossy). >1 = richer conditioning (V11) so the "
                        "model can read input detail and generalize style->atlas.")
    p.add_argument("--encoder", choices=["scratch", "convnext"], default="scratch",
                   help="scratch = from-scratch CNN; convnext = FROZEN pretrained "
                        "ConvNeXt-Tiny pyramid + raw-RGB stream (transferable features "
                        "for cross-skin generalization, V11).")
    p.add_argument("--color-aug", action="store_true",
                   help="V11: paired equivariant color aug — same random gamma/gain/"
                        "bias on BOTH render and target each sample. Breaks fixed-atlas "
                        "memorization, forces reading style from input. Train only.")
    p.add_argument("--style-mod", action="store_true",
                   help="V11: factorize shared structure (learned struct tokens) x "
                        "per-skin style (FiLM-modulated decoder from global encoder "
                        "feats). Targets generalization (vs memorization) across skins.")
    p.add_argument("--decoder", choices=["legacy", "progressive"], default="legacy",
                   help="legacy = single upsample + 2 blocks (small/smooth BMPs); "
                        "progressive = half-res + full-res refine (high-freq detail, "
                        "e.g. EQMAIN slider sprite rows).")
    p.add_argument("--max-minutes", type=float, default=0.0,
                   help="Hard wall-clock cap (minutes). Stop training when exceeded "
                        "(then eval/save run normally). 0 disables. Project rule: "
                        "no training run > 60 min.")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=50)
    p.add_argument("--checkpoint-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", action="store_true",
                   help="Mixed-precision FP16 autocast on CUDA (recommended on T4).")
    p.add_argument("--no-amp", action="store_true",
                   help="Force FP32 on CUDA (overrides --amp).")
    # Adversarial (pix2pix-style) generative training: a PatchGAN discriminator
    # forces crisp high-frequency detail the L1 mean cannot synthesize (EQMAIN
    # slider grooves). Generator/checkpoint unchanged; D is training-only.
    p.add_argument("--adversarial", action="store_true",
                   help="Add a PatchGAN discriminator + adversarial/feature-match loss.")
    p.add_argument("--adv-weight", type=float, default=0.1, help="Weight on the generator adversarial term.")
    p.add_argument("--fm-weight", type=float, default=1.0, help="Weight on discriminator feature-matching loss.")
    p.add_argument("--d-lr", type=float, default=4e-4, help="Discriminator LR (TTUR; > generator LR).")
    p.add_argument("--d-base", type=int, default=64, help="PatchGAN base channels.")
    p.add_argument("--d-layers", type=int, default=3, help="PatchGAN downsample layers.")
    p.add_argument("--init-from", default=None,
                   help="Initialize the generator from a checkpoint (same arch). For "
                        "adversarial: pretrain with L1 first, then fine-tune from it — "
                        "GAN from random weights diverges.")
    # Periodic eval + early stop. One-skin overfit: eval-subset MAE/hit5 is the
    # true gate signal (less noisy than the running-mean train loss). Stops the
    # long flat refinement tail once the gate is comfortably met.
    p.add_argument("--eval-every", type=int, default=0,
                   help="Run a no-grad eval pass every N steps (0 disables). "
                        "Enables early stop and writes eval_progress.jsonl.")
    p.add_argument("--eval-max-items", type=int, default=256,
                   help="Cap eval-subset size (seeded shuffle across families) for speed.")
    p.add_argument("--early-stop", action="store_true",
                   help="Stop once the eval gate holds for --early-stop-patience evals.")
    p.add_argument("--early-stop-mae", type=float, default=0.008,
                   help="Eval MAE threshold for early stop (stricter than the 0.01 pass gate).")
    p.add_argument("--early-stop-hit5", type=float, default=0.93,
                   help="Eval hit_5_255 threshold for early stop (above the 0.90 pass gate).")
    p.add_argument("--early-stop-patience", type=int, default=2,
                   help="Consecutive passing evals required before stopping.")
    args = p.parse_args()

    spec = SPEC_BY_NAME.get(args.bmp)
    if spec is None:
        raise SystemExit(f"--bmp {args.bmp!r} not in TRAINABLE_EXPORT_SPECS")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    ds = BMPExpertDataset(args.data, args.bmp, color_aug=args.color_aug)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=False,
                        num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    model = BMPExpertNet(target_h=spec.h, target_w=spec.w, base=args.base,
                         attn_dim=args.attn_dim, dec_ch=args.dec_ch,
                         heads=args.heads, attn_layers=args.attn_layers,
                         query_div=args.query_div, decoder_kind=args.decoder,
                         kv_scale=args.kv_scale, style_mod=args.style_mod,
                         encoder=args.encoder).to(device)
    if args.init_from:
        from safetensors.torch import load_file as _load
        init_state = _load(args.init_from)
        missing, unexpected = model.load_state_dict(init_state, strict=False)
        allowed = {"query_div_buf", "decoder_kind_buf"}
        if unexpected or set(missing) - allowed:
            raise SystemExit(f"--init-from mismatch: missing={missing} unexpected={unexpected}")
        print(f"initialized generator from {args.init_from}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    disc = d_opt = None
    if args.adversarial:
        disc = BMPPatchDiscriminator(base=args.d_base, n_layers=args.d_layers,
                                     min_dim=min(spec.h, spec.w)).to(device)
        d_opt = torch.optim.AdamW(disc.parameters(), lr=args.d_lr, betas=(0.5, 0.9),
                                  weight_decay=args.weight_decay)

    # Periodic-eval subset: seeded shuffle so it spans all state families (the
    # dataset is ordered by family, so first-N would be unrepresentative).
    eval_loader = None
    if args.eval_every > 0:
        g = torch.Generator().manual_seed(args.seed + 1)
        perm = torch.randperm(len(ds), generator=g).tolist()
        eval_idx = perm[: min(args.eval_max_items, len(ds))]
        eval_loader = DataLoader(Subset(ds, eval_idx), batch_size=args.batch,
                                 shuffle=False, num_workers=args.num_workers,
                                 pin_memory=(device.type == "cuda"))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({
        "bmp": args.bmp, "target_h": spec.h, "target_w": spec.w,
        "n_items": len(ds), "args": vars(args)}, indent=2, sort_keys=True))
    metrics_f = (out / "metrics.jsonl").open("w")
    eval_f = (out / "eval_progress.jsonl").open("w") if args.eval_every > 0 else None

    # Adversarial training runs in FP32 (GAN + fp16 GradScaler is fragile).
    use_amp = device.type == "cuda" and args.amp and not args.no_amp and not args.adversarial
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    autocast_ctx = (lambda: torch.amp.autocast(device_type="cuda", dtype=torch.float16)) \
        if use_amp else contextlib.nullcontext

    print(f"V10 train: bmp={args.bmp} ({spec.w}x{spec.h}) steps={args.steps} "
          f"batch={args.batch} lr={args.lr} n_items={len(ds)} device={device} "
          f"amp={use_amp} progress_every={args.progress_every}", flush=True)

    # Save an initial checkpoint so subsequent eval/infer has artifacts even if
    # the training loop crashes (OOM, etc.) before the first step succeeds.
    save_state_dict(out / "last.safetensors", model.state_dict())
    save_state_dict(out / "best.safetensors", model.state_dict())

    if args.eval_every > 0:
        print(f"periodic eval: every {args.eval_every} steps over "
              f"{len(eval_loader.dataset)} items; early_stop={args.early_stop} "
              f"(mae<{args.early_stop_mae} hit5>{args.early_stop_hit5} "
              f"patience={args.early_stop_patience})", flush=True)

    model.train()
    step = 0
    best = float("inf")
    best_eval_mae = float("inf")
    pass_streak = 0
    stopped_early = False
    recent: list[float] = []
    recent_mae: list[float] = []
    recent_hit: list[float] = []
    start = _time.monotonic()
    last_t = start
    it = iter(loader)
    while step < args.steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        x = batch["render"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)
        if disc is not None:
            # --- Discriminator step (hinge): real target vs detached fake ---
            out_y = model(x)
            d_real, _ = disc(y)
            d_fake, _ = disc(out_y.detach())
            d_loss = F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()
            d_opt.zero_grad(set_to_none=True)
            d_loss.backward()
            d_opt.step()
            # --- Generator step: pixel losses + adversarial + feature matching ---
            losses = compute_losses(out_y, y)
            g_fake, feats_fake = disc(out_y)
            _, feats_real = disc(y)
            g_adv = -g_fake.mean()
            fm = sum(F.l1_loss(ff, fr.detach()) for ff, fr in zip(feats_fake, feats_real)) \
                / max(1, len(feats_fake))
            total = losses["total"] + args.adv_weight * g_adv + args.fm_weight * fm
            losses["total"] = total
            losses["g_adv"] = g_adv.detach()
            losses["d_loss"] = d_loss.detach()
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
        else:
            with autocast_ctx():
                out_y = model(x)
                losses = compute_losses(out_y, y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(losses["total"]).backward()
            scaler.step(optimizer)
            scaler.update()
        step += 1
        recent.append(float(losses["total"].detach()))
        recent_mae.append(float(losses["mae"]))
        recent_hit.append(float(losses["hit_5_255"]))
        if len(recent) > args.progress_every:
            recent.pop(0); recent_mae.pop(0); recent_hit.pop(0)
        rec = {k: float(v.detach() if torch.is_tensor(v) else v) for k, v in losses.items()}
        rec["step"] = step
        metrics_f.write(json.dumps(rec) + "\n")
        # best.safetensors tracks best TRAIN loss only when periodic eval is off;
        # with eval on, the eval block owns best.safetensors (best eval MAE).
        if rec["total"] < best:
            best = rec["total"]
            if eval_loader is None:
                save_state_dict(out / "best.safetensors", model.state_dict())
        if (step % args.checkpoint_every == 0) or step == args.steps:
            save_state_dict(out / "last.safetensors", model.state_dict())
        if args.progress_every > 0 and step % args.progress_every == 0:
            now = _time.monotonic()
            sps = (now - last_t) / args.progress_every
            eta = sps * max(args.steps - step, 0) / 60.0
            print(f"[step {step:>6d}/{args.steps}  {100.0 * step / args.steps:5.1f}%]  "
                  f"loss(mean{len(recent):>4d})={sum(recent) / len(recent):.4f}  "
                  f"mae={sum(recent_mae) / len(recent_mae):.4f}  "
                  f"hit5={sum(recent_hit) / len(recent_hit):.3f}  "
                  f"sec/step={sps:.3f}  elapsed={(now - start) / 60.0:5.1f}min  ETA={eta:5.1f}min",
                  flush=True)
            if disc is not None and "g_adv" in losses:
                print(f"           adv: g_adv={float(losses['g_adv']):.4f} "
                      f"d_loss={float(losses['d_loss']):.4f}", flush=True)
            last_t = now
        # Periodic eval + early stop (true gate signal on a fixed subset).
        if eval_loader is not None and (step % args.eval_every == 0 or step == args.steps):
            e_mae, e_hit = evaluate_subset(model, eval_loader, device, autocast_ctx)
            gate = (e_mae < args.early_stop_mae and e_hit > args.early_stop_hit5)
            pass_streak = pass_streak + 1 if gate else 0
            eval_f.write(json.dumps({"step": step, "eval_mae": e_mae, "eval_hit_5_255": e_hit,
                                     "gate": gate, "pass_streak": pass_streak}) + "\n")
            eval_f.flush()
            if e_mae < best_eval_mae:
                best_eval_mae = e_mae
                save_state_dict(out / "best.safetensors", model.state_dict())
            print(f"[eval @ step {step:>6d}]  eval_mae={e_mae:.5f}  eval_hit5={e_hit:.4f}  "
                  f"gate={'PASS' if gate else 'fail'}  streak={pass_streak}/{args.early_stop_patience}",
                  flush=True)
            if args.early_stop and pass_streak >= args.early_stop_patience:
                save_state_dict(out / "last.safetensors", model.state_dict())
                stopped_early = True
                print(f"EARLY STOP at step {step}: eval gate held {pass_streak} consecutive "
                      f"evals (mae={e_mae:.5f}<{args.early_stop_mae}, "
                      f"hit5={e_hit:.4f}>{args.early_stop_hit5})", flush=True)
                break
        # Hard wall-clock cap (project rule: no training run > ~60 min).
        if args.max_minutes > 0 and (_time.monotonic() - start) / 60.0 >= args.max_minutes:
            save_state_dict(out / "last.safetensors", model.state_dict())
            print(f"TIME CAP at step {step}: hit --max-minutes {args.max_minutes} "
                  f"(elapsed {(_time.monotonic() - start) / 60.0:.1f}min)", flush=True)
            break
    save_state_dict(out / "last.safetensors", model.state_dict())
    metrics_f.close()
    if eval_f is not None:
        eval_f.close()
    print(f"trained {args.bmp} for {step} step(s); best total {best:.4f}; "
          f"best_eval_mae {best_eval_mae:.5f}; early_stopped={stopped_early}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
