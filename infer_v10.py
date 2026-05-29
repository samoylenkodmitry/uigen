#!/usr/bin/env python3
"""V10 inference: mockup -> 11 BMP experts -> skin.wsz -> real Cranamp render.

For each trainable BMP the script loads the matching expert checkpoint (from a
checkpoint directory whose layout mirrors `runs/v10/<bmp_stem>/last.safetensors`)
and predicts the BMP from the input mockup. Missing experts fall back to the
default skin's BMP (so the pipeline always produces a loadable .wsz). The skin
is packaged via save_exported_tensors, including a mockup-derived PLEDIT.TXT.
The deterministic Cranamp render is invoked as the product gate.

Usage:

    python infer_v10.py --image eval_mockups/caat.png \
        --checkpoints runs/v10 --out runs/v10/infer_caat
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from atlas_ai.dataset_v10_bmp import CANVAS_H, CANVAS_W, _image_to_tensor
from atlas_ai.export_spec import TRAINABLE_EXPORT_SPECS
from atlas_ai.v8_assets import save_exported_tensors, tensor_to_image
from atlas_ai.v8_layout import (
    default_layout,
    demo_render_params,
    normalize_mockup_image,
    product_render_params,
)
from atlas_ai.visible_extractor import playlist_pledit_text
from models.bmp_expert_net import BMPExpertNet


CRANAMP_CLI = REPO_ROOT / "cranamp_cli" / "cranamp-cli"


def _load_state_dict(path: Path) -> dict:
    from safetensors.torch import load_file
    return load_file(str(path))


def _build_expert_from_state(state: dict) -> BMPExpertNet:
    g = lambda k: int(state[k].reshape(-1)[0].item())
    return BMPExpertNet(
        target_h=g("target_h_buf"), target_w=g("target_w_buf"),
        base=g("base_buf"), attn_dim=g("attn_dim_buf"),
        dec_ch=g("dec_ch_buf"), heads=g("heads_buf"),
        attn_layers=g("attn_layers_buf"),
        query_div=g("query_div_buf") if "query_div_buf" in state else 4,
        decoder_kind=("progressive" if ("decoder_kind_buf" in state and g("decoder_kind_buf")==1) else "legacy"),
    )


def _predict_bmp(checkpoint: Path, render_t: torch.Tensor, device) -> torch.Tensor:
    state = _load_state_dict(checkpoint)
    model = _build_expert_from_state(state).to(device).eval()
    # Pre-buffer checkpoints (trained before query_div_buf/decoder_kind_buf) omit
    # those keys; the constructor already set them from the inferred defaults, so
    # tolerate exactly those missing buffers (and nothing else).
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed = {"query_div_buf", "decoder_kind_buf"}
    if unexpected or set(missing) - allowed:
        raise RuntimeError(f"checkpoint mismatch: missing={missing} unexpected={unexpected}")
    with torch.no_grad():
        pred = model(render_t.unsqueeze(0).to(device))[0].cpu().clamp(0.0, 1.0)
    return pred


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", required=True, help="Mockup image (any size).")
    ap.add_argument("--checkpoints", default="runs/v10",
                    help="Directory with per-BMP runs (subdir <BMP_STEM>/last.safetensors).")
    ap.add_argument("--out", required=True, help="Output run dir.")
    ap.add_argument("--default-skin", default="assets/default_skin")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-render", action="store_true",
                    help="Skip the real-Cranamp render step (e.g., for tests).")
    ap.add_argument("--demo-state", action="store_true",
                    help="Render with varied (deterministic) EQ/volume/balance/posbar "
                         "positions so the slider sprites are exercised across their "
                         "range (default neutral hides the per-frame picture).")
    args = ap.parse_args()

    device = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Normalize input to the V10 training canvas (960x1728).
    with Image.open(args.image) as im:
        normalized, _scale, _offset = normalize_mockup_image(im, size=(CANVAS_W, CANVAS_H))
    normalized.save(out / "normalized.png")
    render_path_for_loader = out / "normalized.png"
    render_t = _image_to_tensor(render_path_for_loader, size=(CANVAS_W, CANVAS_H))

    # Run each expert if its checkpoint exists; else fall back to the default BMP.
    ck_root = Path(args.checkpoints)
    files: dict[str, torch.Tensor] = {}
    used: dict[str, str] = {}
    for spec in TRAINABLE_EXPORT_SPECS:
        stem = Path(spec.file_name).stem
        ck = ck_root / stem / "last.safetensors"
        if ck.exists():
            pred = _predict_bmp(ck, render_t, device)
            if pred.shape != (3, spec.h, spec.w):
                raise SystemExit(f"{spec.file_name}: expected (3,{spec.h},{spec.w}), got {tuple(pred.shape)}")
            files[spec.file_name] = pred
            used[spec.file_name] = str(ck)
        else:
            used[spec.file_name] = f"DEFAULT ({args.default_skin})"
    (out / "experts_used.json").write_text(json.dumps(used, indent=2))

    # PLEDIT.TXT colors from the mockup (deployable; engine will honor them).
    layout = default_layout(CANVAS_W, CANVAS_H)
    pledit_txt = playlist_pledit_text(normalized, layout, default_skin=args.default_skin)

    # Package skin.wsz (missing BMPs fill from default).
    skin_dir = out / "skin"
    zip_path = save_exported_tensors(
        files, skin_dir, default_skin=args.default_skin,
        package=True, text_overrides={"PLEDIT.TXT": pledit_txt},
    )
    print(f"wrote {zip_path}", flush=True)

    # Also save the predicted BMPs as PNGs alongside, for inspection.
    pred_dir = out / "predicted_bmps"
    pred_dir.mkdir(exist_ok=True)
    for name, tensor in files.items():
        tensor_to_image(tensor).save(pred_dir / f"{Path(name).stem}.png")

    if args.no_render:
        return 0

    # Deterministic real-Cranamp render of the generated .wsz.
    render_params_json = out / "render_params.json"
    rparams = demo_render_params(layout) if args.demo_state else product_render_params(layout)
    render_params_json.write_text(json.dumps(rparams, indent=2))
    render_cranamp = out / "render_cranamp.png"
    res = subprocess.run(
        [str(CRANAMP_CLI), "render-params",
         "--skin-dir", str(skin_dir), "--params-json", str(render_params_json),
         "--canvas-w", str(CANVAS_W), "--canvas-h", str(CANVAS_H),
         "--out-view", str(render_cranamp)],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=180,
    )
    if res.returncode != 0:
        print(f"Cranamp render FAILED: rc={res.returncode}\n{res.stderr[-500:]}", flush=True)
        return res.returncode
    # Side-by-side: mockup | real Cranamp render.
    cr = Image.open(render_cranamp).convert("RGB")
    nb = normalized.convert("RGB")
    sbs = Image.new("RGB", (nb.width + cr.width, max(nb.height, cr.height)), (8, 8, 10))
    sbs.paste(nb, (0, 0))
    sbs.paste(cr, (nb.width, 0))
    sbs.save(out / "side_by_side.png")
    print(f"wrote {out / 'side_by_side.png'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
