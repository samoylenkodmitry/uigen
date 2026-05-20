#!/usr/bin/env python3
"""Mockup image -> SlotNet skin files -> classic skin.wsz."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from atlas_ai.export import export_atlas_to_skin
from atlas_ai.export_spec import blank_atlas_like_files
from atlas_ai.profiles import load_atlas_profile, load_export_profile
from models.slotnet_v34 import SlotNetV34
from models.slotnet_v35 import SlotNetV35
from models.slotnet_v5 import SlotNetV5


INPUT_H = 1728
INPUT_W = 960


@dataclass(frozen=True)
class CheckpointInfo:
    version: int
    base_channels: int
    style_dim: int | None = None
    head_channels: int | None = None
    attn_dim: int | None = None
    attention_heads: int | None = None
    cross_attention_layers: int | None = None
    file_embedding_dim: int | None = None
    frequencies: tuple[int, ...] | None = None


def _info_from_state(get) -> CheckpointInfo:
    """Build a CheckpointInfo using a tensor accessor (file or dict)."""
    version = int(get("slotnet_version").reshape(-1)[0].item())
    base_channels = int(get("enc1.0.weight").shape[0])
    if version == 35:
        style_dim = int(get("style_proj.0.weight").shape[0])
        head_channels = int(get("heads.main.body.0.weight").shape[0])
        return CheckpointInfo(version, base_channels, style_dim, head_channels)
    if version == 50:
        style_dim = int(get("style_proj.0.weight").shape[0])
        attn_dim = int(get("feature_proj.weight").shape[0])
        attention_heads = int(get("attention_heads_buffer").reshape(-1)[0].item())
        cross_attention_layers = int(get("cross_attention_layers_buffer").reshape(-1)[0].item())
        file_embedding_dim = int(get("file_embedding.weight").shape[1])
        head_channels = int(get("heads.main.decoder.0.weight").shape[0])
        try:
            frequencies = tuple(int(x) for x in get("frequencies_buffer").reshape(-1).tolist())
        except Exception:
            frequencies = None
        return CheckpointInfo(
            version,
            base_channels,
            style_dim=style_dim,
            head_channels=head_channels,
            attn_dim=attn_dim,
            attention_heads=attention_heads,
            cross_attention_layers=cross_attention_layers,
            file_embedding_dim=file_embedding_dim,
            frequencies=frequencies,
        )
    return CheckpointInfo(version, base_channels)


def detect_checkpoint_info(ckpt_path: Path) -> CheckpointInfo:
    try:
        from safetensors.torch import safe_open
        with safe_open(str(ckpt_path), framework="pt", device="cpu") as f:
            return _info_from_state(f.get_tensor)
    except Exception:
        state = torch.load(str(ckpt_path), map_location="cpu")
        return _info_from_state(state.__getitem__)


def build_model_from_info(info: CheckpointInfo, *, device: torch.device, atlas_profile=None):
    """Construct the right SlotNet variant given a checkpoint info."""
    if info.version == 34:
        if atlas_profile is None:
            raise ValueError("V3.4 dispatch needs atlas_profile")
        return SlotNetV34(atlas_profile=atlas_profile, base_channels=info.base_channels).to(device)
    if info.version == 35:
        return SlotNetV35(
            base_channels=info.base_channels,
            style_dim=info.style_dim or 192,
            head_channels=info.head_channels,
        ).to(device)
    if info.version == 50:
        kwargs = {
            "base_channels": info.base_channels,
            "style_dim": info.style_dim or 192,
            "head_channels": info.head_channels,
            "attn_dim": info.attn_dim or 128,
            "attention_heads": info.attention_heads or 4,
            "cross_attention_layers": info.cross_attention_layers or 1,
            "file_embedding_dim": info.file_embedding_dim or 32,
        }
        if info.frequencies:
            kwargs["frequencies"] = info.frequencies
        return SlotNetV5(**kwargs).to(device)
    raise ValueError(f"unsupported SlotNet version {info.version}")


def detect_base_channels(ckpt_path: Path) -> int:
    return detect_checkpoint_info(ckpt_path).base_channels


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
    parser.add_argument("--slotnet-style-dim", type=int, default=None)
    parser.add_argument("--slotnet-head-channels", type=int, default=None)
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

    ckpt_info = detect_checkpoint_info(Path(args.slotnet))
    if args.slotnet_base_channels or args.slotnet_style_dim or args.slotnet_head_channels:
        ckpt_info = CheckpointInfo(
            version=ckpt_info.version,
            base_channels=args.slotnet_base_channels or ckpt_info.base_channels,
            style_dim=args.slotnet_style_dim or ckpt_info.style_dim,
            head_channels=args.slotnet_head_channels or ckpt_info.head_channels,
            attn_dim=ckpt_info.attn_dim,
            attention_heads=ckpt_info.attention_heads,
            cross_attention_layers=ckpt_info.cross_attention_layers,
            file_embedding_dim=ckpt_info.file_embedding_dim,
        )
    model = build_model_from_info(ckpt_info, device=device, atlas_profile=atlas_profile)
    load_checkpoint(model, Path(args.slotnet))
    model.eval()
    with torch.no_grad():
        output = model(image_to_tensor(canvas, device))
        if ckpt_info.version == 34:
            rgb = output["prediction"][0].sigmoid().clamp(0, 1)
        else:
            files = {name: logits[0].sigmoid().clamp(0, 1) for name, logits in output["files"].items()}
            rgb = blank_atlas_like_files(files)

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
