#!/usr/bin/env python3
"""Mockup image -> V3.4 RGB atlas -> classic skin.wsz."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from atlas_ai.export import export_atlas_to_skin
from atlas_ai.profiles import load_atlas_profile, load_export_profile
from models.slotnet_v34 import SlotNetV34


INPUT_H = 1728
INPUT_W = 960


def detect_base_channels(ckpt_path: Path) -> int:
    try:
        from safetensors.torch import safe_open
        with safe_open(str(ckpt_path), framework="pt", device="cpu") as f:
            version = int(f.get_tensor("slotnet_version").reshape(-1)[0].item())
            if version != 34:
                raise SystemExit(f"checkpoint is SlotNet version {version}, expected 34")
            return int(f.get_tensor("enc1.0.weight").shape[0])
    except SystemExit:
        raise
    except Exception:
        state = torch.load(str(ckpt_path), map_location="cpu")
        version = int(state["slotnet_version"].reshape(-1)[0].item())
        if version != 34:
            raise SystemExit(f"checkpoint is SlotNet version {version}, expected 34")
        return int(state["enc1.0.weight"].shape[0])


def letterbox_to_canvas(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    src_w, src_h = img.size
    scale = min(target_w / src_w, target_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), (14, 14, 18))
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return canvas


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--slotnet", required=True)
    parser.add_argument("--atlas-profile", default="configs/atlas_train_v1.json")
    parser.add_argument("--export-profile", default="configs/export_profile_classic.json")
    parser.add_argument("--default-skin", default="assets/default_skin")
    parser.add_argument("--slotnet-base-channels", type=int, default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for stale_name in ["input_letterboxed.png", "inference_meta.json"]:
        (out / stale_name).unlink(missing_ok=True)
    device = torch.device(args.device)
    atlas_profile = load_atlas_profile(args.atlas_profile)
    export_profile = load_export_profile(args.export_profile)

    with Image.open(args.image) as src:
        canvas = letterbox_to_canvas(src.convert("RGB"), INPUT_W, INPUT_H)

    base_channels = args.slotnet_base_channels or detect_base_channels(Path(args.slotnet))
    model = SlotNetV34(atlas_profile=atlas_profile, base_channels=base_channels).to(device)
    load_checkpoint(model, Path(args.slotnet))
    model.eval()
    with torch.no_grad():
        logits = model(image_to_tensor(canvas, device))["prediction"][0]
        rgb = logits.sigmoid().clamp(0, 1)

    atlas_arr = (rgb.cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    atlas_path = out / "atlas.png"
    Image.fromarray(atlas_arr, "RGB").save(atlas_path)
    zip_path = export_atlas_to_skin(
        atlas_path=atlas_path,
        atlas_profile=atlas_profile,
        export_profile=export_profile,
        default_skin=args.default_skin,
        out_dir=out,
    )
    print(f"wrote {zip_path}")
    print(f"  atlas: {atlas_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
