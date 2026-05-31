#!/usr/bin/env python3
"""V11 ROUND-TRIP product sanity metric (codex consult #5).

Answers "is the predicted atlas product-usable?" without relying on the exact-MAE
or cond_eval proxy. For each held-out skin:

  oracle = render_visible(skin's REAL target BMPs)            # neutral fixed layout
  pred   = render_visible(REAL targets, but CBUTTONS replaced # model conditions on
           by the model's prediction from that skin's render) #   the skin's render

We compare pred vs oracle over the component's VISIBLE rects only (for CBUTTONS:
the 6 transport-button rects in the main window). own = self prediction; shuffled =
a *different* held skin's prediction inserted into this skin -> renders -> compared
to THIS skin's oracle. A real conditional model has own << shuffled.

LOCK (codex): rt_gap = shuffled - own >= +0.08  OR  own <= 0.75 * shuffled, and no
systematic wrong-color/style buttons in the grid. Judged on VISIBLE px, the product
signal — not the hidden-atlas diversity proxy.

Only CBUTTONS is wired now (its visible rects); --bmp guards against misuse.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from atlas_ai.torch_cranamp_renderer import render_visible, _layout_rect  # noqa: E402
from atlas_ai.v8_layout import NORMALIZED_SIZE, default_layout       # noqa: E402
from models.bmp_expert_net import BMPExpertNet                      # noqa: E402

BMP_NAMES = ["MAIN.bmp", "TITLEBAR.bmp", "CBUTTONS.bmp", "PLAYPAUS.bmp", "MONOSTER.bmp",
             "POSBAR.bmp", "VOLUME.bmp", "BALANCE.bmp", "SHUFREP.bmp", "EQMAIN.bmp",
             "PLEDIT.bmp"]

# CBUTTONS transport buttons: (src w,h) and window-local (x,y) in 275x116 main space.
CBUTTONS_BUTTONS = [((23, 18), (16, 88)), ((23, 18), (39, 88)), ((23, 18), (62, 88)),
                    ((23, 18), (85, 88)), ((22, 18), (108, 88)), ((22, 16), (136, 89))]


def _build_expert_from_state(state: dict) -> BMPExpertNet:
    g = lambda k: int(state[k].reshape(-1)[0].item())
    return BMPExpertNet(
        target_h=g("target_h_buf"), target_w=g("target_w_buf"),
        base=g("base_buf"), attn_dim=g("attn_dim_buf"),
        dec_ch=g("dec_ch_buf"), heads=g("heads_buf"), attn_layers=g("attn_layers_buf"),
        query_div=g("query_div_buf") if "query_div_buf" in state else 4,
        kv_scale=g("kv_scale_buf") if "kv_scale_buf" in state else 1,
        style_mod=bool("style_mod_buf" in state and g("style_mod_buf") == 1),
        encoder=("convnext" if ("encoder_buf" in state and g("encoder_buf") == 1) else "scratch"),
        decoder_kind=("progressive" if ("decoder_kind_buf" in state and g("decoder_kind_buf") == 1) else "legacy"),
    )


def _load_img(path: Path) -> torch.Tensor:
    im = Image.open(path).convert("RGB")
    return torch.from_numpy(__import__("numpy").asarray(im)).float().permute(2, 0, 1) / 255.0


def _button_mask(layout: dict, h: int, w: int) -> torch.Tensor:
    x, y, mw, mh = _layout_rect(layout, "main")
    sx, sy = mw / 275.0, mh / 116.0
    mask = torch.zeros(1, h, w)
    for (bw, bh), (lx, ly) in CBUTTONS_BUTTONS:
        bx, by = round(x + lx * sx), round(y + ly * sy)
        ww, hh = max(1, round(bw * sx)), max(1, round(bh * sy))
        mask[:, by:by + hh, bx:bx + ww] = 1.0
    return mask


def _masked_mae(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> float:
    m3 = mask.expand_as(a)
    return float((a - b).abs().mul(m3).sum() / m3.sum().clamp_min(1.0))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="Held-out dataset dir (targets/<skin>/*.bmp + renders/ + csv/).")
    ap.add_argument("--bmp", default="CBUTTONS.bmp")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--baseline-checkpoint", default="", help="Optional: compare own-MAE vs a baseline ckpt.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--canvas-w", type=int, default=NORMALIZED_SIZE[0])
    args = ap.parse_args()
    if args.bmp != "CBUTTONS.bmp":
        raise SystemExit(f"round-trip visible rects only wired for CBUTTONS.bmp (got {args.bmp})")

    device = torch.device(args.device)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import load_file

    def load_model(ckpt: str) -> BMPExpertNet:
        p = Path(ckpt)
        if not p.exists():
            p = p.with_name("last.safetensors")
        st = load_file(str(p))
        m = _build_expert_from_state(st)
        m.load_state_dict(st, strict=False)
        return m.to(device).eval()

    model = load_model(args.checkpoint)
    base_model = load_model(args.baseline_checkpoint) if args.baseline_checkpoint else None

    data = Path(args.data)
    rows = list(csv.DictReader((data / "csv" / "train_CBUTTONS.csv").open()))
    # one representative render per skin (first variant seen)
    skin_render: dict[str, str] = {}
    for r in rows:
        skin_render.setdefault(r["skin_id"], r["render_png"])
    skins = sorted(skin_render)
    print(f"round-trip: {len(skins)} held skins, bmp={args.bmp}, device={device}", flush=True)

    canvas_w = args.canvas_w
    canvas_h = round(canvas_w / NORMALIZED_SIZE[0] * NORMALIZED_SIZE[1])
    layout = default_layout(canvas_w, canvas_h)

    @torch.no_grad()
    def predict(m: BMPExpertNet, skin: str) -> torch.Tensor:
        rend = _load_img(data / skin_render[skin]).unsqueeze(0).to(device)
        return m(rend)[0].clamp(0, 1).cpu()       # [3,h,w] canonical CBUTTONS

    # cache real target file dicts + oracle renders + predictions
    real_files: dict[str, dict] = {}
    oracle: dict[str, torch.Tensor] = {}
    preds: dict[str, torch.Tensor] = {}
    base_preds: dict[str, torch.Tensor] = {}
    for s in skins:
        files = {n: _load_img(data / "targets" / s / n) for n in BMP_NAMES}
        real_files[s] = files
        oracle[s] = render_visible(files, layout)
        preds[s] = predict(model, s)
        if base_model is not None:
            base_preds[s] = predict(base_model, s)

    mask = _button_mask(layout, canvas_h, canvas_w)

    def render_with(skin: str, cbuttons: torch.Tensor) -> torch.Tensor:
        files = dict(real_files[skin]); files["CBUTTONS.bmp"] = cbuttons
        return render_visible(files, layout)

    own_maes, shuf_maes, base_own = [], [], []
    per_skin = {}
    for i, s in enumerate(skins):
        own = _masked_mae(render_with(s, preds[s]), oracle[s], mask)
        j = skins[(i + 1) % len(skins)]                       # a different skin's prediction
        shuf = _masked_mae(render_with(s, preds[j]), oracle[s], mask)
        own_maes.append(own); shuf_maes.append(shuf)
        rec = {"own": round(own, 5), "shuffled": round(shuf, 5)}
        if base_model is not None:
            b = _masked_mae(render_with(s, base_preds[s]), oracle[s], mask)
            base_own.append(b); rec["base_own"] = round(b, 5)
        per_skin[s] = rec

    own_m = sum(own_maes) / len(own_maes)
    shuf_m = sum(shuf_maes) / len(shuf_maes)
    gap = shuf_m - own_m
    res = {"bmp": args.bmp, "n_skins": len(skins),
           "own_visible_mae": round(own_m, 5), "shuffled_visible_mae": round(shuf_m, 5),
           "rt_gap": round(gap, 5), "own_over_shuffled": round(own_m / max(shuf_m, 1e-6), 4),
           "lock_pass": bool(gap >= 0.08 or own_m <= 0.75 * shuf_m), "per_skin": per_skin}
    if base_model is not None:
        bm = sum(base_own) / len(base_own)
        res["base_own_visible_mae"] = round(bm, 5)
        res["cond_disc_vs_base_delta"] = round(own_m - bm, 5)  # <=.01 worse = ok
    (out / "roundtrip.json").write_text(json.dumps(res, indent=2))

    # grid: oracle-buttons | pred-buttons for first 8 skins (crop of button band)
    x, y, mw, mh = _layout_rect(layout, "main")
    sy = mh / 116.0
    by0 = round(y + 86 * sy); by1 = round(y + 108 * sy)
    bx0 = round(x + 10 * (mw / 275.0)); bx1 = round(x + 162 * (mw / 275.0))
    tiles = []
    for s in skins[:8]:
        o = oracle[s][:, by0:by1, bx0:bx1]; p = render_with(s, preds[s])[:, by0:by1, bx0:bx1]
        tiles.append(torch.cat([o, p], dim=2))
    if tiles:
        grid = torch.cat(tiles, dim=1).clamp(0, 1)
        Image.fromarray((grid.permute(1, 2, 0) * 255).byte().numpy()).save(out / "roundtrip_grid.png")

    print(f">>> ROUNDTRIP {args.bmp} n={len(skins)}: own_vis={own_m:.4f} shuf_vis={shuf_m:.4f} "
          f"rt_gap={gap:+.4f} own/shuf={own_m / max(shuf_m, 1e-6):.3f} "
          f"LOCK_PASS={res['lock_pass']}"
          + (f" base_own={res['base_own_visible_mae']:.4f} delta={res['cond_disc_vs_base_delta']:+.4f}"
             if base_model is not None else ""), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
