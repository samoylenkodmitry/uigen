#!/usr/bin/env python3
"""Refine exported BMP pixels against a normalized mockup render loss."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from atlas_ai.torch_cranamp_renderer import render_visible
from atlas_ai.v8_assets import image_to_tensor, load_exported_tensors, save_exported_tensors, tensor_to_image
from atlas_ai.v8_layout import load_layout


def _inv_sigmoid(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(1e-4, 1.0 - 1e-4)
    return torch.log(x / (1.0 - x))


def _sobel(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        x = x.unsqueeze(0)
    kx = x.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
    ky = x.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)
    kernels_x = kx.repeat(x.shape[1], 1, 1, 1)
    kernels_y = ky.repeat(x.shape[1], 1, 1, 1)
    gx = F.conv2d(x, kernels_x, padding=1, groups=x.shape[1])
    gy = F.conv2d(x, kernels_y, padding=1, groups=x.shape[1])
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, help="Normalized mockup image.")
    parser.add_argument("--layout-json", required=True)
    parser.add_argument("--skin-dir", required=True, help="Initial exported BMP directory.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--edge-weight", type=float, default=0.25)
    parser.add_argument("--prior-weight", type=float, default=0.04)
    parser.add_argument("--default-skin", default="assets/default_skin")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    layout = load_layout(args.layout_json)
    with Image.open(args.target) as image:
        target = image_to_tensor(image.resize(tuple(layout["normalized_size"]), Image.Resampling.LANCZOS)).to(device)
    init = {k: v.to(device) for k, v in load_exported_tensors(args.skin_dir, default_skin=args.default_skin).items()}
    params = {k: torch.nn.Parameter(_inv_sigmoid(v)) for k, v in init.items()}
    opt = torch.optim.Adam(params.values(), lr=args.lr)
    target_edge = _sobel(target)

    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        files = {k: torch.sigmoid(v) for k, v in params.items()}
        rendered = render_visible(files, layout)
        rgb = (rendered - target).abs().mean()
        edge = (_sobel(rendered) - target_edge).abs().mean()
        prior = torch.stack([(files[k] - init[k]).abs().mean() for k in files]).mean()
        loss = rgb + args.edge_weight * edge + args.prior_weight * prior
        loss.backward()
        opt.step()
        if step == 0 or (step + 1) % 25 == 0 or step + 1 == args.steps:
            print(
                f"[{step + 1:04d}/{args.steps}] loss={float(loss.detach()):.5f} "
                f"rgb={float(rgb.detach()):.5f} edge={float(edge.detach()):.5f} "
                f"prior={float(prior.detach()):.5f}",
                flush=True,
            )

    final = {k: torch.sigmoid(v.detach()).cpu() for k, v in params.items()}
    out = Path(args.out)
    zip_path = save_exported_tensors(final, out / "skin", default_skin=args.default_skin, package=True)
    tensor_to_image(render_visible(final, layout)).save(out / "render_preview.png")
    print(f"wrote {zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
