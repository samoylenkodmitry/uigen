#!/usr/bin/env python3
"""V6 Stage 2 evaluation.

Loads a SlotNetV6 checkpoint and reports the Stage 2 acceptance metrics:

    visible MAE          mean |grid_sample(view, uv_pred) - clean target|
                         at visible source pixels (one number per file, plus
                         aggregate).
    copy_conf AUC        ROC-AUC of sigmoid(copy_conf_logits) against
                         visible_mask. Per file and aggregate.
    UV median error      median |uv_pred - uv_target|_pixels at visible source
                         pixels, expressed in *input view pixels*. Per file
                         and aggregate.

Stage 2 passes when, on the training CSV:

    aggregate visible_mae    < 0.01
    aggregate copy_conf_auc  > 0.98
    aggregate uv_median_px   < 2.0

Run:

    python scripts/17_eval_slotnet_v6.py \
        --samples data_v6_default_skin/train.csv \
        --skin-source assets/default_skin \
        --slotnet runs/slotnet_v6_default_skin/snapshot_stepNNNNNN.safetensors \
        --device cuda \
        --out-json runs/slotnet_v6_default_skin/eval_stepNNNNNN.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from atlas_ai.dataset_v6 import V6CopyDataset
from atlas_ai.export_spec import TRAINABLE_EXPORT_SPECS
from models.losses_v6 import grid_sample_copy, refine_copy_rgb
from models.slotnet_v6 import SlotNetV6


def load_state_dict(path: Path) -> dict:
    from safetensors.torch import load_file
    return load_file(str(path))


def _detect_v6_kwargs(state: dict) -> dict:
    version = int(state["slotnet_version"].reshape(-1)[0].item())
    if version != 60:
        raise SystemExit(f"need a V6 checkpoint (version 60), got {version}")
    return {
        "base_channels": int(state["enc1.0.weight"].shape[0]),
        "style_dim": int(state["style_proj.0.weight"].shape[0]),
        "head_channels": int(state["heads.main.body.0.weight"].shape[0]),
        "attn_dim": int(state["attn_dim_buffer"].reshape(-1)[0].item()),
        "attention_heads": int(state["attention_heads_buffer"].reshape(-1)[0].item()),
        "cross_attention_layers": int(state["cross_attention_layers_buffer"].reshape(-1)[0].item()),
        "file_embedding_dim": int(state["file_embedding.weight"].shape[1]),
        "query_grid_divisor": int(state["query_grid_divisor_buffer"].reshape(-1)[0].item()),
    }


def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC via the Mann-Whitney U formulation. Handles ties."""
    if scores.size == 0 or labels.sum() == 0 or labels.sum() == labels.size:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    n = order.size
    i = 0
    # Average ranks for ties (mergesort gives a stable order; tie groups are
    # contiguous after argsort).
    sorted_scores = scores[order]
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg = 0.5 * (i + j) + 1.0  # ranks are 1-based
        ranks[order[i : j + 1]] = avg
        i = j + 1
    pos_mask = labels.astype(bool)
    n_pos = int(pos_mask.sum())
    n_neg = int(n - n_pos)
    sum_pos_ranks = float(ranks[pos_mask].sum())
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _uv_pixel_error(
    uv_pred: torch.Tensor,
    uv_target: torch.Tensor,
    visible: torch.Tensor,
    view_h: int,
    view_w: int,
) -> torch.Tensor:
    """L2 distance |uv_pred - uv_target| converted to input view pixels.

    UV is in [-1, 1] under align_corners=False, so pixel = (uv + 1) * size / 2.
    Therefore |du|_pixels = |du| * W / 2 and |dv|_pixels = |dv| * H / 2.
    """
    if visible.dim() == 3:
        visible = visible.unsqueeze(1)
    du = uv_pred[:, 0] - uv_target[:, 0]
    dv = uv_pred[:, 1] - uv_target[:, 1]
    dx = du * view_w * 0.5
    dy = dv * view_h * 0.5
    err = torch.sqrt(dx * dx + dy * dy)  # [B, H, W]
    visible_bhw = visible.squeeze(1)
    return err[visible_bhw > 0]


def evaluate(
    model: SlotNetV6,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Compute per-file and aggregate Stage 2 metrics across a DataLoader."""
    model.eval()
    per_file_mae: dict[str, list[float]] = {spec.file_name: [] for spec in TRAINABLE_EXPORT_SPECS}
    per_file_mae_n: dict[str, list[int]] = {spec.file_name: [] for spec in TRAINABLE_EXPORT_SPECS}
    per_file_uv_err: dict[str, list[torch.Tensor]] = {spec.file_name: [] for spec in TRAINABLE_EXPORT_SPECS}
    per_file_scores: dict[str, list[np.ndarray]] = {spec.file_name: [] for spec in TRAINABLE_EXPORT_SPECS}
    per_file_labels: dict[str, list[np.ndarray]] = {spec.file_name: [] for spec in TRAINABLE_EXPORT_SPECS}
    with torch.no_grad():
        for batch in loader:
            view = batch["view"].to(device)
            view_h, view_w = view.shape[-2:]
            output = model(view)
            files = output["files"]
            for spec in TRAINABLE_EXPORT_SPECS:
                head_out = files[spec.file_name]
                uv_pred = head_out["uv"]
                conf_logits = head_out["conf_logits"]
                residual = head_out.get("residual")
                target = batch["files"][spec.file_name]["target"].to(device)
                visible = batch["files"][spec.file_name]["visible"].to(device)
                uv_target = batch["files"][spec.file_name]["uv"].to(device)

                # visible MAE via the production copy_refined path: bilinear
                # grid_sample (+ residual correction if the head has one).
                copy_rgb = grid_sample_copy(view, uv_pred)
                copy_rgb = refine_copy_rgb(copy_rgb, residual)
                mask3 = visible.expand_as(copy_rgb)
                diff = (copy_rgb - target).abs() * mask3
                n_vis = float(mask3.sum().item())
                if n_vis > 0:
                    per_file_mae[spec.file_name].append(float(diff.sum().item()) / n_vis)
                    per_file_mae_n[spec.file_name].append(int(n_vis / 3))

                # UV pixel error at visible source pixels.
                err = _uv_pixel_error(uv_pred, uv_target, visible.squeeze(1), view_h, view_w)
                if err.numel():
                    per_file_uv_err[spec.file_name].append(err.detach().cpu())

                # Conf AUC: predicted score (sigmoid) vs visible label.
                scores = torch.sigmoid(conf_logits).squeeze(1).reshape(-1).detach().cpu().numpy()
                labels = visible.squeeze(1).reshape(-1).detach().cpu().numpy()
                per_file_scores[spec.file_name].append(scores)
                per_file_labels[spec.file_name].append(labels)

    per_file_summary: dict[str, dict[str, float]] = {}
    visible_mae_pixels = 0.0
    visible_mae_count = 0
    aggregate_uv: list[float] = []
    for spec in TRAINABLE_EXPORT_SPECS:
        maes = per_file_mae[spec.file_name]
        ns = per_file_mae_n[spec.file_name]
        if maes:
            total_diff = sum(m * n for m, n in zip(maes, ns))
            total_n = sum(ns)
            mae = total_diff / max(1, total_n)
            visible_mae_pixels += total_diff
            visible_mae_count += total_n
        else:
            mae = float("nan")
        errs = (torch.cat(per_file_uv_err[spec.file_name]).numpy()
                if per_file_uv_err[spec.file_name] else np.zeros(0, dtype=np.float32))
        uv_median = float(np.median(errs)) if errs.size else float("nan")
        if errs.size:
            aggregate_uv.append(errs)
        scores = np.concatenate(per_file_scores[spec.file_name]) if per_file_scores[spec.file_name] else np.zeros(0)
        labels = np.concatenate(per_file_labels[spec.file_name]) if per_file_labels[spec.file_name] else np.zeros(0)
        auc = _roc_auc(scores, labels)
        per_file_summary[spec.file_name] = {
            "visible_mae": mae,
            "uv_median_px": uv_median,
            "copy_conf_auc": auc,
            "visible_pixels": float(sum(ns)),
        }

    agg_uv_errs = np.concatenate(aggregate_uv) if aggregate_uv else np.zeros(0, dtype=np.float32)
    aggregate_scores = np.concatenate(
        [np.concatenate(per_file_scores[s.file_name]) for s in TRAINABLE_EXPORT_SPECS
         if per_file_scores[s.file_name]]
    ) if any(per_file_scores[s.file_name] for s in TRAINABLE_EXPORT_SPECS) else np.zeros(0)
    aggregate_labels = np.concatenate(
        [np.concatenate(per_file_labels[s.file_name]) for s in TRAINABLE_EXPORT_SPECS
         if per_file_labels[s.file_name]]
    ) if any(per_file_labels[s.file_name] for s in TRAINABLE_EXPORT_SPECS) else np.zeros(0)
    return {
        "aggregate": {
            "visible_mae": (
                visible_mae_pixels / max(1, visible_mae_count)
                if visible_mae_count else float("nan")
            ),
            "uv_median_px": float(np.median(agg_uv_errs)) if agg_uv_errs.size else float("nan"),
            "copy_conf_auc": _roc_auc(aggregate_scores, aggregate_labels),
        },
        "per_file": per_file_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--skin-source", required=True)
    parser.add_argument("--slotnet", required=True)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    dataset = V6CopyDataset(args.samples, args.skin_source)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=0)

    state = load_state_dict(Path(args.slotnet))
    kwargs = _detect_v6_kwargs(state)
    model = SlotNetV6(**kwargs).to(device)
    model.load_state_dict(state)

    result = evaluate(model, loader, device)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
