#!/usr/bin/env python3
"""Dump target / prediction / |target-pred| panels for a V7 completer
checkpoint on a chosen subset of (skin, file) tuples.

When the strip / BV probes stall around a particular skin or file, the
per-file / per-skin scalar metrics tell us *which* tuples are stuck but
not *how* the model fails: is it shifted hue, smeared edges, hallucinated
content, or a frame the state-family mask doesn't actually pin? This
script answers that by saving side-by-side images for visual inspection.

Outputs one PNG per (skin, file, mask_seed) at <out-dir>/<skin>__<file>__
seed<NN>.png. Layout: 3 panels horizontally — target, prediction (with
observed-mask overlay outline), abs-diff *5 (clipped to 1).

Usage:
    python scripts/dump_v7_completer_diffs.py \\
        --state-families configs/state_families_classic.yaml \\
        --skin-sources 'skin1=/path1,skin2=/path2' \\
        --checkpoint runs/v7_completer_xxx/last.safetensors \\
        --files BALANCE.bmp,VOLUME.bmp \\
        --skins goodgawd_bba84deb,a_halo_so_bright_it_bleeds \\
        --mask-mode state_family \\
        --num-seeds 3 \\
        --out-dir diffs/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from atlas_ai.dataset_v7_completion import V7CompletionDataset
from atlas_ai.export_spec import TRAINABLE_EXPORT_SPECS
from atlas_ai.support_mask import load_support_masks
from atlas_ai.v7_masks import V7MaskWeights
from models.v7_completer import V7Completer


FILE_TO_ID: dict[str, int] = {spec.file_name: idx for idx, spec in enumerate(TRAINABLE_EXPORT_SPECS)}


def _parse_skin_sources(spec: str) -> dict[str, str]:
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
    if not out:
        raise SystemExit(f"--skin-sources must declare at least one skin (got {spec!r})")
    return out


def _load_checkpoint(path: Path) -> tuple[V7Completer, dict]:
    from safetensors.torch import load_file
    state = load_file(str(path))
    version = int(state["model_version"].reshape(-1)[0].item())
    if version not in (70, 71):
        raise SystemExit(f"need V7 completer checkpoint (version 70 or 71), got {version}")
    kwargs = {
        "base_channels": int(state["base_channels_buffer"].reshape(-1)[0].item()),
        "file_embedding_dim": int(state["file_embedding_dim_buffer"].reshape(-1)[0].item()),
    }
    if "num_skins_buffer" in state:
        kwargs["num_skins"] = int(state["num_skins_buffer"].reshape(-1)[0].item())
    if "skin_embedding_dim_buffer" in state:
        kwargs["skin_embedding_dim"] = int(state["skin_embedding_dim_buffer"].reshape(-1)[0].item())
    model = V7Completer(**kwargs)
    model.load_state_dict(state, strict=False)
    return model, state


def _mask_weights_for(mode: str) -> V7MaskWeights:
    if mode == "state_family":
        return V7MaskWeights(provenance=0.0, state_family=1.0,
                             random_rect=0.0, whole_file=0.0, passthrough=0.0)
    if mode == "random_rect":
        return V7MaskWeights(provenance=0.0, state_family=0.0,
                             random_rect=1.0, whole_file=0.0, passthrough=0.0)
    if mode == "provenance":
        return V7MaskWeights(provenance=1.0, state_family=0.0,
                             random_rect=0.0, whole_file=0.0, passthrough=0.0)
    raise SystemExit(f"--mask-mode must be state_family|random_rect|provenance, got {mode!r}")


def _to_uint8(tensor: torch.Tensor) -> np.ndarray:
    """[3, H, W] float in [0, 1] -> [H, W, 3] uint8."""
    arr = tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return (arr * 255.0 + 0.5).astype(np.uint8)


def _diff_panel(target: torch.Tensor, pred: torch.Tensor, *, gain: float = 5.0) -> np.ndarray:
    diff = (pred - target).abs() * gain
    return _to_uint8(diff)


def _mask_outline(mask: torch.Tensor, color=(255, 0, 0)) -> np.ndarray | None:
    """Return an [H, W, 3] uint8 outline of the *boundary* of the observed
    mask, transparent (zeros) elsewhere. Used to overlay on the pred panel
    so we see what the model was told vs what it had to invent."""
    m = mask.detach().cpu().numpy()
    if m.ndim == 3:
        m = m[0]
    m = (m > 0.5).astype(np.uint8)
    if m.sum() == 0:
        return None
    # Cheap boundary: pixel is on boundary if it's 1 and any 4-neighbor is 0
    up = np.pad(m, ((1, 0), (0, 0)), mode="edge")[:-1]
    down = np.pad(m, ((0, 1), (0, 0)), mode="edge")[1:]
    left = np.pad(m, ((0, 0), (1, 0)), mode="edge")[:, :-1]
    right = np.pad(m, ((0, 0), (0, 1)), mode="edge")[:, 1:]
    boundary = m & ~(up & down & left & right)
    out = np.zeros((m.shape[0], m.shape[1], 3), dtype=np.uint8)
    out[boundary == 1] = color
    return out


def _save_panel(path: Path, panels: list[np.ndarray], gap: int = 4, bg: int = 32) -> None:
    """Concatenate [H, W, 3] uint8 panels horizontally with `gap` px filler."""
    from PIL import Image
    if not panels:
        return
    h = max(p.shape[0] for p in panels)
    padded: list[np.ndarray] = []
    for p in panels:
        if p.shape[0] < h:
            pad = np.full((h - p.shape[0], p.shape[1], 3), bg, dtype=np.uint8)
            p = np.concatenate([p, pad], axis=0)
        padded.append(p)
    gap_strip = np.full((h, gap, 3), bg, dtype=np.uint8)
    pieces: list[np.ndarray] = []
    for i, p in enumerate(padded):
        if i:
            pieces.append(gap_strip)
        pieces.append(p)
    full = np.concatenate(pieces, axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(full).save(str(path))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-families", required=True)
    parser.add_argument("--skin-sources", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--files", default="",
                        help="Comma-separated file names to dump; empty = all.")
    parser.add_argument("--skins", default="",
                        help="Comma-separated skin ids to dump; empty = all.")
    parser.add_argument("--mask-mode", choices=["state_family", "random_rect", "provenance"],
                        default="state_family")
    parser.add_argument("--num-seeds", type=int, default=3,
                        help="Number of mask RNG seeds to sample per (skin, file).")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    skin_sources = _parse_skin_sources(args.skin_sources)
    model, _ = _load_checkpoint(Path(args.checkpoint))
    model.to(device).eval()

    files_filter = {f.strip() for f in args.files.split(",") if f.strip()}
    skins_filter = {s.strip() for s in args.skins.split(",") if s.strip()}

    dataset = V7CompletionDataset(
        skin_sources=skin_sources,
        state_families_path=args.state_families,
        mask_weights=_mask_weights_for(args.mask_mode),
        seed=0,
    )
    support_masks = load_support_masks()

    items: list[tuple[int, str, str]] = []
    for idx, (skin_id, file_name) in enumerate(dataset.items):
        if files_filter and file_name not in files_filter:
            continue
        if skins_filter and skin_id not in skins_filter:
            continue
        items.append((idx, skin_id, file_name))
    if not items:
        raise SystemExit("no items match --files / --skins filters")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"dumping {len(items)} (skin, file) tuples * {args.num_seeds} seeds "
          f"= {len(items) * args.num_seeds} panels to {out_dir}")

    with torch.no_grad():
        for idx, skin_id, file_name in items:
            for seed in range(args.num_seeds):
                dataset.set_epoch(seed)
                item = dataset[idx]
                observed_rgb = item["observed_rgb"].unsqueeze(0).to(device)
                observed_mask = item["observed_mask"].unsqueeze(0).to(device)
                target_rgb = item["target_rgb"].unsqueeze(0).to(device)
                file_id = torch.tensor([FILE_TO_ID[file_name]], dtype=torch.long, device=device)
                skin_id_tensor = None
                if model.num_skins > 0:
                    si = item["skin_index"]
                    if not isinstance(si, torch.Tensor):
                        si = torch.tensor(si, dtype=torch.long)
                    skin_id_tensor = si.reshape(1).to(device)
                pred = model(observed_rgb, observed_mask, file_id, skin_id=skin_id_tensor)
                support = support_masks[file_name].to(device=device, dtype=pred.dtype)
                if support.dim() == 2:
                    support = support.unsqueeze(0).unsqueeze(0)
                support3 = support.expand_as(pred)
                # Restrict displays to support pixels: outside support is
                # padding the renderer never reads.
                t_disp = (target_rgb * support3).squeeze(0)
                p_disp = (pred.clamp(0, 1) * support3).squeeze(0)
                diff_disp = ((pred - target_rgb).abs() * support3 * 5.0).clamp(0, 1).squeeze(0)
                tgt_panel = _to_uint8(t_disp)
                pred_panel = _to_uint8(p_disp)
                outline = _mask_outline(observed_mask.squeeze(0).squeeze(0))
                if outline is not None:
                    nonzero = outline.any(axis=-1)
                    pred_panel[nonzero] = outline[nonzero]
                diff_panel = _to_uint8(diff_disp)
                out_path = out_dir / f"{skin_id}__{file_name.replace('.', '_')}__seed{seed:02d}.png"
                _save_panel(out_path, [tgt_panel, pred_panel, diff_panel])
    print(f"wrote {len(items) * args.num_seeds} panels to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
