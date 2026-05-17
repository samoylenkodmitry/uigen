#!/usr/bin/env python3
"""Mockup image -> packed atlas -> classic Winamp .wsz.

SlotNetV3 pipeline (full-image -> full-atlas, no GeoNet dependency in
SlotNet code path):

  1. Letterbox input to (INPUT_H, INPUT_W) = (1728, 960).
  2. SlotNetV3 -> [1, 7, 1024, 1024] full atlas in one forward.
  3. Apply magenta snap per slot (only for slots whose policy enables it).
  4. Save the predicted atlas; export_atlas_to_skin crops slots -> BMPs -> .wsz.

GeoNet output (rects + state) is still emitted as a side debug artifact
(rects_overlay.png and rects_pred.f32) for inspection, but no SlotNet code
path depends on it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))

from atlas_ai.export import export_atlas_to_skin
from atlas_ai.profiles import load_atlas_profile, load_export_profile, load_json
from atlas_ai.rects import derive_eq_band_rects
from models.atlas import pack_default_atlas_tensor
from models.geonet80 import GeoNet80, decode_rects
from models.slotnet_v31 import SlotNetV31
from models.slotnet_v32 import SlotNetV32


def detect_slotnet_config(ckpt_path: Path) -> tuple[str, int]:
    """Return (model_version, base_channels) by inspecting checkpoint tensors.

    V3.2 checkpoints include `observed_head.*` and `residual_enabled_mask`;
    V3.1 does not. The base_channels count is recovered from the first
    encoder conv: `enc1.0.weight` has shape `[base_channels, 3, 3, 3]`.

    Both pieces of info are required for a strict checkpoint load -- if
    the user trained with `--base-channels 16` but inference instantiates
    with the default 24, the load fails with a shape mismatch. Auto-
    detection makes the inference path robust to either training setting.
    """
    try:
        from safetensors.torch import safe_open
        with safe_open(str(ckpt_path), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            base_channels = int(f.get_tensor("enc1.0.weight").shape[0])
    except Exception:
        state = torch.load(str(ckpt_path), map_location="cpu")
        keys = list(state.keys())
        base_channels = int(state["enc1.0.weight"].shape[0])
    if any(k.startswith("observed_head") or k == "residual_enabled_mask" for k in keys):
        return "v32", base_channels
    return "v31", base_channels


def detect_slotnet_model(ckpt_path: Path) -> str:
    """Backwards-compat wrapper that returns only the model version."""
    return detect_slotnet_config(ckpt_path)[0]

INPUT_H = 1728
INPUT_W = 960


def letterbox_to_canvas(img: Image.Image, target_w: int, target_h: int) -> tuple[Image.Image, dict]:
    src_w, src_h = img.size
    scale = min(target_w / src_w, target_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), (14, 14, 18))
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, {"src_w": src_w, "src_h": src_h, "scale": scale, "pad_x": pad_x, "pad_y": pad_y}


def image_to_tensor(img: Image.Image, device: torch.device) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous().to(device)


def load_checkpoint(model: torch.nn.Module, ckpt_path: Path) -> None:
    try:
        from safetensors.torch import load_file
        state = load_file(str(ckpt_path))
    except Exception:
        state = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(state)


COMPONENT_LABELS = {
    0: "main_window", 1: "main_titlebar", 5: "main_posbar", 6: "main_transport_row",
    13: "main_volume_block", 15: "main_balance_block",
    17: "main_shuffle_button", 18: "main_repeat_button",
    19: "main_eq_toggle", 20: "main_pl_toggle", 21: "main_mono_stereo",
    22: "main_playpause_indicator", 23: "main_window_buttons",
    24: "eq_window", 25: "eq_titlebar", 27: "eq_on_auto_block",
    28: "eq_preamp_slider", 29: "eq_sliders_group",
    40: "eq_presets_button", 41: "eq_close_button",
    42: "playlist_window", 43: "playlist_titlebar", 44: "playlist_text_area",
    46: "playlist_scrollbar_track", 47: "playlist_scrollbar_thumb",
    48: "playlist_bottom_bar",
}


def draw_rect_overlay(canvas: Image.Image, rects: np.ndarray, save_path: Path) -> None:
    overlay = canvas.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")
    w, h = canvas.size
    for k in range(rects.shape[0]):
        x0n, y0n, x1n, y1n, vis = rects[k]
        if vis <= 0.5:
            continue
        x0, y0, x1, y1 = int(x0n * w), int(y0n * h), int(x1n * w), int(y1n * h)
        if x1 <= x0 or y1 <= y0:
            continue
        color = (255, 80, 80, 255) if k <= 23 else (80, 200, 255, 255) if k <= 41 else (160, 255, 120, 255)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        label = COMPONENT_LABELS.get(k, f"c{k}")
        draw.text((x0 + 2, y0 + 2), label, fill=color)
    overlay.save(save_path)


def apply_magenta_snap(atlas_rgb: torch.Tensor, special_logits: torch.Tensor, atlas_profile, policy: dict) -> torch.Tensor:
    """Per slot, if policy enables magenta, snap pixels where magenta class wins."""
    out = atlas_rgb.clone()
    per_slot = policy.get("per_slot", {})
    default = bool(policy.get("default", False))
    special_probs = special_logits.softmax(dim=0)
    for slot in atlas_profile.slots:
        if not bool(per_slot.get(slot.name, default)):
            continue
        y0, y1 = slot.y, slot.y + slot.h
        x0, x1 = slot.x, slot.x + slot.w
        mag = special_probs[1, y0:y1, x0:x1] > 0.60
        if mag.any():
            out[0, y0:y1, x0:x1][mag] = 1.0
            out[1, y0:y1, x0:x1][mag] = 0.0
            out[2, y0:y1, x0:x1][mag] = 1.0
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--geonet", required=True, help="Used only for the debug rect overlay; not part of SlotNet's code path.")
    parser.add_argument("--slotnet", required=True)
    parser.add_argument("--atlas-profile", default="configs/atlas_v1.json")
    parser.add_argument("--export-profile", default="configs/export_profile_classic.json")
    parser.add_argument("--magenta-policy", default="configs/magenta_policy.json")
    parser.add_argument("--default-skin", default="assets/default_skin")
    parser.add_argument("--geonet-base-channels", type=int, default=32)
    parser.add_argument("--geonet-fpn-channels", type=int, default=96)
    parser.add_argument("--slotnet-base-channels", type=int, default=None,
                        help="Override SlotNet base_channels. Auto-detected from checkpoint shape if omitted.")
    parser.add_argument("--slotnet-model", choices=["v31", "v32"], default=None,
                        help="SlotNet variant. Auto-detected from checkpoint if omitted.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    atlas_profile = load_atlas_profile(args.atlas_profile)
    export_profile = load_export_profile(args.export_profile)
    magenta_policy = load_json(args.magenta_policy)

    with Image.open(args.image) as src:
        img = src.convert("RGB")
    canvas, transform = letterbox_to_canvas(img, INPUT_W, INPUT_H)
    canvas.save(out / "input_letterboxed.png")
    view = image_to_tensor(canvas, device)

    # GeoNet: debug-only overlay. Not used downstream.
    geonet = GeoNet80(base_channels=args.geonet_base_channels, fpn_channels=args.geonet_fpn_channels).to(device)
    load_checkpoint(geonet, Path(args.geonet))
    geonet.eval()
    with torch.no_grad():
        gout = geonet(view, anchor_rects=None, jitter_state_anchors=False)
    rects = decode_rects(gout["heatmap"], gout["wh"], gout["offset"])[0]
    state = gout["state"][0].detach()
    bands = derive_eq_band_rects(tuple(rects[29].cpu().tolist()))
    for idx, band in enumerate(bands):
        rects[30 + idx] = torch.tensor(band, device=rects.device, dtype=rects.dtype)
    rects.cpu().numpy().astype("<f4").tofile(out / "rects_pred.f32")
    state.cpu().numpy().astype("<f4").tofile(out / "state_pred.f32")
    draw_rect_overlay(canvas, rects.cpu().numpy(), out / "rects_overlay.png")

    # SlotNetV3.1/V3.2: full-atlas inference with default-atlas prior + layout
    # conditioning. Both the model variant AND base_channels are auto-detected
    # from the checkpoint -- V3.2 has `observed_head.*`, and base_channels is
    # recovered from `enc1.0.weight.shape[0]`. CLI flags override.
    default_atlas = pack_default_atlas_tensor(args.default_skin, atlas_profile).to(device)
    detected_model, detected_base = detect_slotnet_config(Path(args.slotnet))
    requested_model = args.slotnet_model or detected_model
    requested_base = args.slotnet_base_channels if args.slotnet_base_channels is not None else detected_base
    if requested_model == "v32":
        slotnet = SlotNetV32(atlas_profile=atlas_profile, default_atlas=default_atlas, base_channels=requested_base).to(device)
    elif requested_model == "v31":
        slotnet = SlotNetV31(atlas_profile=atlas_profile, default_atlas=default_atlas, base_channels=requested_base).to(device)
    else:
        raise SystemExit(f"unknown --slotnet-model {requested_model!r}; expected 'v31' or 'v32'")
    print(f"  slotnet: model={requested_model} base_channels={requested_base}")
    load_checkpoint(slotnet, Path(args.slotnet))
    slotnet.eval()
    with torch.no_grad():
        sout = slotnet(view)
        logits = sout["prediction"][0]                # [7, H, W]
        rgb = logits[:3].sigmoid().clamp(0, 1)        # [3, H, W]
        special_logits = logits[3:7]                  # [4, H, W]
        rgb = apply_magenta_snap(rgb, special_logits, atlas_profile, magenta_policy)
        # Save the prior for debugging too -- this is what the model sees as
        # its starting point (default skin recolored to input statistics).
        prior_arr = (sout["prior_rgb"][0].cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(prior_arr, "RGB").save(out / "_prior.png")

    atlas_arr = (rgb.cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    atlas_img = Image.fromarray(atlas_arr, "RGB")
    atlas_path = out / "_predicted_atlas.png"
    atlas_img.save(atlas_path)

    zip_path = export_atlas_to_skin(
        atlas_path=atlas_path,
        atlas_profile=atlas_profile,
        export_profile=export_profile,
        default_skin=args.default_skin,
        out_dir=out,
    )
    atlas_path.unlink(missing_ok=True)

    with (out / "inference_meta.json").open("w", encoding="utf-8") as f:
        json.dump({
            "input": str(args.image),
            "transform": transform,
            "visible_rects": int((rects[:, 4] > 0.5).sum().item()),
            "geonet": str(args.geonet),
            "slotnet": str(args.slotnet),
            "wsz": str(zip_path),
        }, f, indent=2, sort_keys=True)

    print(f"wrote {zip_path}")
    print(f"  atlas:        {out / 'atlas.png'}")
    print(f"  rect overlay: {out / 'rects_overlay.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
