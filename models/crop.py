from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from atlas_ai.rects import encode_rect


def crop_view_regions(
    images: torch.Tensor,
    rects: torch.Tensor,
    output_hw: tuple[int, int],
    input_hw: tuple[int, int] = (1672, 941),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Crop normalized rects from `images` with grid_sample.

    `rects` is `[B,5]` with normalized `[x0,y0,x1,y1,visible]`.
    Returns `(crop, log_scale, valid)` where log_scale is `[B,2]`.
    """
    if images.ndim != 4:
        raise ValueError("images must be BxCxHxW")
    if rects.ndim != 2 or rects.shape[1] != 5:
        raise ValueError("rects must be Bx5")
    out_h, out_w = output_hw
    batch = images.shape[0]
    device = images.device
    dtype = images.dtype

    ys = (torch.arange(out_h, device=device, dtype=dtype) + 0.5) / out_h
    xs = (torch.arange(out_w, device=device, dtype=dtype) + 0.5) / out_w
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    x0, y0, x1, y1, visible = rects.to(device=device, dtype=dtype).unbind(dim=1)
    width = (x1 - x0).clamp_min(0.0)
    height = (y1 - y0).clamp_min(0.0)
    valid = (visible > 0.0) & (width > 0.0) & (height > 0.0)

    grid_x = x0[:, None, None] + xx[None, :, :] * width[:, None, None]
    grid_y = y0[:, None, None] + yy[None, :, :] * height[:, None, None]
    grid = torch.stack((grid_x * 2.0 - 1.0, grid_y * 2.0 - 1.0), dim=-1)
    crops = F.grid_sample(images, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    crops = crops * valid.to(dtype)[:, None, None, None]

    input_h, input_w = input_hw
    render_w = width * float(input_w)
    render_h = height * float(input_h)
    scale_x = torch.where(valid, render_w / float(out_w), torch.ones_like(render_w))
    scale_y = torch.where(valid, render_h / float(out_h), torch.ones_like(render_h))
    log_scale = torch.stack((scale_x.clamp_min(1e-6).log(), scale_y.clamp_min(1e-6).log()), dim=1)
    log_scale = torch.where(valid[:, None], log_scale, torch.zeros_like(log_scale))
    return crops, log_scale, valid


def jitter_rects(rects: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    x0, y0, x1, y1, visible = rects.unbind(dim=-1)
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    w = (x1 - x0).clamp_min(1e-6)
    h = (y1 - y0).clamp_min(1e-6)
    rand = lambda shape: torch.rand(shape, device=rects.device, dtype=rects.dtype, generator=generator)
    cx = cx + (rand(cx.shape) * 2.0 - 1.0) * 0.08 * w
    cy = cy + (rand(cy.shape) * 2.0 - 1.0) * 0.08 * h
    w = w * torch.exp((rand(w.shape) * 2.0 - 1.0) * 0.12)
    h = h * torch.exp((rand(h.shape) * 2.0 - 1.0) * 0.12)
    out = torch.stack(
        (
            (cx - w * 0.5).clamp(0.0, 1.0),
            (cy - h * 0.5).clamp(0.0, 1.0),
            (cx + w * 0.5).clamp(0.0, 1.0),
            (cy + h * 0.5).clamp(0.0, 1.0),
            visible,
        ),
        dim=-1,
    )
    invisible = visible <= 0.0
    out[invisible] = rects[invisible]
    return out


def coordinate_channels(batch: int, height: int, width: int, device=None, dtype=None) -> torch.Tensor:
    ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((xx, yy), dim=0).expand(batch, -1, -1, -1)


__all__ = ["encode_rect", "crop_view_regions", "jitter_rects", "coordinate_channels"]
