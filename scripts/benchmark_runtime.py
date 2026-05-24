#!/usr/bin/env python3
"""Benchmark a V7 completer training step on the current runtime.

Drives a short training loop (default 200 steps) with the model + dataset
that an experiment YAML specifies, then prints:

    GPU / device / AMP setting
    seconds per step (median over the last 80% of steps)
    peak CUDA VRAM (when applicable)
    estimated wall time for the experiment's full --steps

Does not write checkpoints. Intentionally minimal so it runs in seconds
on CPU and in well under a minute on GPU.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from atlas_ai.dataset_v7_completion import V7CompletionDataset
from atlas_ai.export_spec import TRAINABLE_EXPORT_SPECS
from atlas_ai.support_mask import load_support_masks
from atlas_ai.v7_batching import SameFileBatchSampler, WeightedSameFileBatchSampler
from atlas_ai.v7_masks import V7MaskWeights
from models.losses_v7 import support_masked_l1_loss, support_masked_sobel_mae
from models.v7_completer import V7Completer
from scripts.print_env import collect_env
from scripts.run_experiment import build_command, _format_skin_sources
from scripts.runtime_config import load_runtime_from_env_or_arg


FILE_TO_ID = {spec.file_name: idx for idx, spec in enumerate(TRAINABLE_EXPORT_SPECS)}


def _parse_skin_sources_arg(arg: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for entry in arg.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if "=" in entry:
            sid, p = entry.split("=", 1)
            out[sid.strip()] = p.strip()
        else:
            p = Path(entry)
            out[p.name] = str(p)
    return out


def _extract_arg(argv: list[str], flag: str, default=None):
    if flag in argv:
        i = argv.index(flag)
        if i + 1 < len(argv):
            return argv[i + 1]
    return default


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", default=None)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--steps", type=int, default=200,
                        help="Benchmark step count. Default 200; 2-5 is enough for a CPU smoke.")
    parser.add_argument("--batch", type=int, default=None,
                        help="Override batch size; defaults to experiment's batch.")
    parser.add_argument("--device", default=None,
                        help="Override device; defaults to runtime's device.")
    parser.add_argument("--within-file-replacement", dest="within_file_replacement",
                        type=lambda s: str(s).lower() not in ("0", "false", "no"),
                        default=None,
                        help="Weighted-mode only. Mirror of the trainer flag so the sweep "
                             "exercises the same code path. If omitted, reads the experiment "
                             "YAML's value (default True).")
    args = parser.parse_args()

    runtime = load_runtime_from_env_or_arg(args.runtime)
    experiment_path = Path(args.experiment).resolve()
    experiment = yaml.safe_load(experiment_path.read_text(encoding="utf-8")) or {}
    _, manifest = build_command(experiment, runtime)
    argv: list[str] = manifest["argv"]

    skin_sources_arg = _extract_arg(argv, "--skin-sources")
    if not skin_sources_arg:
        raise SystemExit("experiment did not yield a --skin-sources argument")
    state_families = _extract_arg(argv, "--state-families")
    if not state_families:
        raise SystemExit("experiment did not yield --state-families")

    base_channels = int(_extract_arg(argv, "--base-channels", 24))
    file_embedding_dim = int(_extract_arg(argv, "--file-embedding-dim", 32))
    skin_embedding_dim = int(_extract_arg(argv, "--skin-embedding-dim", 0))
    sobel_weight = float(_extract_arg(argv, "--sobel-weight", 0.0))
    sampling_mode = _extract_arg(argv, "--sampling-mode", "epoch")
    full_steps = int(_extract_arg(argv, "--steps", 1))
    batch_size = int(args.batch or _extract_arg(argv, "--batch", 1))
    if args.within_file_replacement is None:
        wfr_raw = _extract_arg(argv, "--within-file-replacement", "true")
        within_file_replacement = str(wfr_raw).lower() not in ("0", "false", "no")
    else:
        within_file_replacement = bool(args.within_file_replacement)

    device = torch.device(args.device or runtime.device)
    skin_sources = _parse_skin_sources_arg(skin_sources_arg)

    print("env:")
    env = collect_env()
    print(f"  python:    {env.get('python_version')}")
    print(f"  torch:     {env.get('torch_version')}")
    print(f"  device:    {device}")
    print(f"  amp:       {runtime.amp}")
    print(f"  batch:     {batch_size}")
    print(f"  sampling:  {sampling_mode} (within_file_replacement={within_file_replacement})")
    if env.get("cuda_devices"):
        d = env["cuda_devices"][0]
        print(f"  GPU:       {d['name']} ({d['total_memory_mib']} MiB)")
    print()

    dataset = V7CompletionDataset(
        skin_sources=skin_sources,
        state_families_path=state_families,
        mask_weights=V7MaskWeights(),
    )
    if sampling_mode == "weighted":
        # Even weights for benchmark (don't read the YAML; bench is structural).
        weights = {spec.file_name: 1.0 for spec in TRAINABLE_EXPORT_SPECS}
        sampler = WeightedSameFileBatchSampler(
            dataset.items, batch_size=batch_size, file_weights=weights,
            num_batches=args.steps, generator=torch.Generator().manual_seed(0),
            within_file_replacement=within_file_replacement,
        )
    else:
        sampler = SameFileBatchSampler(
            dataset.items, batch_size=batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(0),
        )
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
    support_masks = load_support_masks()

    num_skins = len(dataset.skin_ids) if skin_embedding_dim > 0 else 0
    model = V7Completer(
        base_channels=base_channels, file_embedding_dim=file_embedding_dim,
        num_skins=num_skins, skin_embedding_dim=skin_embedding_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    use_amp = runtime.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    times: list[float] = []
    model.train()
    seen = 0
    for batch in loader:
        if seen >= args.steps:
            break
        t0 = time.perf_counter()
        view = batch["observed_rgb"].to(device, non_blocking=True)
        mask_t = batch["observed_mask"].to(device, non_blocking=True)
        target = batch["target_rgb"].to(device, non_blocking=True)
        file_name = batch["file_name"][0] if isinstance(batch["file_name"], list) else batch["file_name"]
        file_id = torch.full((view.shape[0],), FILE_TO_ID[file_name], dtype=torch.long, device=device)
        skin_id_tensor = None
        if model.num_skins > 0:
            skin_idx = batch.get("skin_index")
            if not isinstance(skin_idx, torch.Tensor):
                skin_idx = torch.as_tensor(skin_idx, dtype=torch.long)
            skin_id_tensor = skin_idx.to(device=device, dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            final = model(view, mask_t, file_id, skin_id=skin_id_tensor)
            support = support_masks[file_name].to(device=device, dtype=final.dtype)
            loss = support_masked_l1_loss(final, target, support)
            if sobel_weight > 0:
                loss = loss + sobel_weight * support_masked_sobel_mae(final, target, support)
        if use_amp:
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            loss.backward(); optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append(time.perf_counter() - t0)
        seen += 1

    if not times:
        raise SystemExit("benchmark produced 0 steps")

    # Use the last 80% to skip warmup.
    keep = max(1, int(len(times) * 0.8))
    body = times[-keep:]
    median = statistics.median(body)
    p90 = statistics.quantiles(body, n=10)[-1] if len(body) >= 10 else max(body)
    peak_mib = (torch.cuda.max_memory_allocated(device) // (1024 * 1024)) if device.type == "cuda" else None

    print(f"steps run:        {len(times)} (warmup excluded: last {keep})")
    print(f"batch size:       {batch_size}")
    print(f"sec/step median:  {median:.4f}")
    print(f"sec/step p90:     {p90:.4f}")
    if peak_mib is not None:
        print(f"peak VRAM:        {peak_mib} MiB")
    print(f"estimated full {full_steps} steps: {median * full_steps:.1f} s "
          f"({median * full_steps / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
