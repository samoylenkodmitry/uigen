"""Smoke tests for V7 completer trainer + eval."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from atlas_ai.export_spec import TRAINABLE_EXPORT_FILES


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "train_v7_completer.py"
EVAL_SCRIPT = ROOT / "scripts/19_eval_v7_completer.py"
DEFAULT_SKIN = ROOT / "assets/default_skin"
CONFIG = ROOT / "configs/state_families_classic.yaml"


def _trainer_args(out: Path, *extras: str) -> list[str]:
    return [
        sys.executable, str(TRAINER),
        "--state-families", str(CONFIG),
        "--skin-sources", f"default={DEFAULT_SKIN}",
        "--batch", "2",
        "--base-channels", "8",
        "--file-embedding-dim", "8",
        "--num-workers", "0",
        "--checkpoint-every", "1",
        "--seed", "123",
        "--device", "cpu",
        "--out", str(out),
        *extras,
    ]


def test_trainer_smoke_runs_and_writes_artifacts(tmp_path):
    run = tmp_path / "v7_run"
    subprocess.run(_trainer_args(run, "--steps", "3"), check=True)
    assert (run / "config.json").exists()
    assert (run / "metrics.jsonl").exists()
    assert (run / "best.safetensors").exists()
    assert (run / "last.safetensors").exists()
    lines = (run / "metrics.jsonl").read_text().strip().splitlines()
    assert len(lines) == 3
    for line in lines:
        entry = json.loads(line)
        assert {"step", "epoch", "total", "l1", "file_name"} <= set(entry)


def test_trainer_loss_drops_over_few_steps(tmp_path):
    """Sanity that the model actually trains.

    Keep the sampled file and mask family fixed; otherwise this smoke test
    measures sampler/mask difficulty variance rather than optimization.
    """
    run = tmp_path / "v7_run"
    subprocess.run(_trainer_args(
        run,
        "--steps", "30",
        "--batch", "1",
        "--only-file", "PLAYPAUS.bmp",
        "--mask-provenance", "0",
        "--mask-state-family", "0",
        "--mask-random-rect", "0",
        "--mask-whole-file", "1",
        "--lr", "0.01",
    ), check=True)
    metrics = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines()]
    first_avg = sum(m["total"] for m in metrics[:5]) / 5.0
    last_avg = sum(m["total"] for m in metrics[-5:]) / 5.0
    assert last_avg < first_avg, f"loss did not drop: first_avg={first_avg} last_avg={last_avg}"


def test_eval_reloads_checkpoint_and_reports_metrics(tmp_path):
    run = tmp_path / "v7_run"
    subprocess.run(_trainer_args(run, "--steps", "2"), check=True)
    eval_json = run / "eval.json"
    subprocess.run(
        [
            sys.executable, str(EVAL_SCRIPT),
            "--state-families", str(CONFIG),
            "--skin-sources", f"default={DEFAULT_SKIN}",
            "--checkpoint", str(run / "last.safetensors"),
            "--batch", "2",
            "--mask-samples", "2",
            "--device", "cpu",
            "--out-json", str(eval_json),
        ],
        check=True,
    )
    result = json.loads(eval_json.read_text())
    agg = result["aggregate"]
    assert {"supported_mae", "hit5", "sobel_mae", "mask_samples"} <= set(agg)
    assert agg["mask_samples"] == 2
    assert set(result["per_file"]) == set(TRAINABLE_EXPORT_FILES)
    for file_name in TRAINABLE_EXPORT_FILES:
        entry = result["per_file"][file_name]
        assert {"supported_mae", "hit5", "sobel_mae", "samples"} <= set(entry)
    # Per-skin breakdown: with one skin id ("default"), there should be
    # exactly one entry and its shape mirrors the per-file shape.
    assert set(result["per_skin"]) == {"default"}
    skin_entry = result["per_skin"]["default"]
    assert {"supported_mae", "hit5", "sobel_mae", "samples"} <= set(skin_entry)
    assert skin_entry["samples"] > 0


def test_trainer_rejects_non_v7_checkpoint_at_eval(tmp_path):
    """Eval script should fail clearly if the checkpoint is not V7."""
    from safetensors.torch import save_file
    import torch as _torch
    bogus = tmp_path / "bogus.safetensors"
    save_file({"model_version": _torch.tensor([99], dtype=_torch.int32)}, str(bogus))
    result = subprocess.run(
        [
            sys.executable, str(EVAL_SCRIPT),
            "--state-families", str(CONFIG),
            "--skin-sources", f"default={DEFAULT_SKIN}",
            "--checkpoint", str(bogus),
            "--device", "cpu",
        ],
        capture_output=True,
    )
    assert result.returncode != 0
    assert b"V7 completer" in result.stderr or b"V7 completer" in result.stdout


def test_trainer_skin_sources_basename_default(tmp_path):
    """Bare-path skin-sources uses basename as skin id."""
    run = tmp_path / "v7_run"
    subprocess.run(
        [
            sys.executable, str(TRAINER),
            "--state-families", str(CONFIG),
            "--skin-sources", str(DEFAULT_SKIN),  # bare path
            "--batch", "1",
            "--base-channels", "8",
            "--file-embedding-dim", "8",
            "--num-workers", "0",
            "--checkpoint-every", "1",
            "--device", "cpu",
            "--out", str(run),
            "--steps", "1",
        ],
        check=True,
    )
    config = json.loads((run / "config.json").read_text())
    assert "default_skin" in config["skin_sources"]
