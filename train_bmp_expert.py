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
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from atlas_ai.dataset_v10_bmp import BMPExpertDataset
from atlas_ai.export_spec import TRAINABLE_EXPORT_SPECS
from models.bmp_expert_net import BMPExpertNet


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
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=50)
    p.add_argument("--checkpoint-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", action="store_true",
                   help="Mixed-precision FP16 autocast on CUDA (recommended on T4).")
    p.add_argument("--no-amp", action="store_true",
                   help="Force FP32 on CUDA (overrides --amp).")
    args = p.parse_args()

    spec = SPEC_BY_NAME.get(args.bmp)
    if spec is None:
        raise SystemExit(f"--bmp {args.bmp!r} not in TRAINABLE_EXPORT_SPECS")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    ds = BMPExpertDataset(args.data, args.bmp)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=False,
                        num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    model = BMPExpertNet(target_h=spec.h, target_w=spec.w, base=args.base,
                         attn_dim=args.attn_dim, dec_ch=args.dec_ch,
                         heads=args.heads, attn_layers=args.attn_layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({
        "bmp": args.bmp, "target_h": spec.h, "target_w": spec.w,
        "n_items": len(ds), "args": vars(args)}, indent=2, sort_keys=True))
    metrics_f = (out / "metrics.jsonl").open("w")

    use_amp = device.type == "cuda" and args.amp and not args.no_amp
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

    model.train()
    step = 0
    best = float("inf")
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
        if rec["total"] < best:
            best = rec["total"]
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
            last_t = now
    save_state_dict(out / "last.safetensors", model.state_dict())
    metrics_f.close()
    print(f"trained {args.bmp} for {step} step(s); best total {best:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
