"""Smoke tests for the V6 Stage 2 trainer and eval scripts.

These tests do not assert acceptance metrics; they assert that the trainer can
take a few optimizer steps without exceptions and produce a checkpoint that
the eval script reloads and scores.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
DATASET_SCRIPT = ROOT / "scripts/16_make_v6_dataset.py"
TRAINER = ROOT / "train_slotnet_v6.py"
EVAL_SCRIPT = ROOT / "scripts/17_eval_slotnet_v6.py"


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("v6_data")
    subprocess.run(
        [
            sys.executable, str(DATASET_SCRIPT),
            "--skin-source", str(ROOT / "assets/default_skin"),
            "--skin-id", "default_skin_test",
            "--variants", "2",
            "--canvas-w", "240",
            "--canvas-h", "432",
            "--out", str(out),
        ],
        check=True,
    )
    return out


def test_trainer_runs_a_few_steps_and_writes_checkpoint(tiny_dataset, tmp_path):
    run_dir = tmp_path / "v6_run"
    subprocess.run(
        [
            sys.executable, str(TRAINER),
            "--train", str(tiny_dataset / "train.csv"),
            "--skin-source", str(ROOT / "assets/default_skin"),
            "--steps", "3",
            "--batch", "1",
            "--base-channels", "8",
            "--style-dim", "32",
            "--head-channels", "16",
            "--attn-dim", "32",
            "--attention-heads", "4",
            "--cross-attention-layers", "1",
            "--file-embedding-dim", "8",
            "--query-grid-divisor", "4",
            "--num-workers", "0",
            "--checkpoint-every", "1",
            "--snapshot-every", "0",
            "--out", str(run_dir),
            "--device", "cpu",
        ],
        check=True,
    )
    last = run_dir / "last.safetensors"
    best = run_dir / "best.safetensors"
    metrics = run_dir / "metrics.jsonl"
    config = run_dir / "config.json"
    assert last.exists()
    assert best.exists()
    assert metrics.exists()
    assert config.exists()
    # Three optimizer steps -> three lines logged.
    lines = metrics.read_text().strip().splitlines()
    assert len(lines) == 3
    parsed = [json.loads(line) for line in lines]
    for entry in parsed:
        assert set(entry) == {"step", "total", "copy_rgb", "conf", "uv", "uv_tv"}


def test_trainer_resume_from_loads_weights(tiny_dataset, tmp_path):
    """--resume-from seeds the model from a prior V6 checkpoint and continues."""
    run_a = tmp_path / "phase_a"
    common = [
        sys.executable, str(TRAINER),
        "--train", str(tiny_dataset / "train.csv"),
        "--skin-source", str(ROOT / "assets/default_skin"),
        "--batch", "1",
        "--base-channels", "8",
        "--style-dim", "32",
        "--head-channels", "16",
        "--attn-dim", "32",
        "--attention-heads", "4",
        "--cross-attention-layers", "1",
        "--file-embedding-dim", "8",
        "--query-grid-divisor", "4",
        "--num-workers", "0",
        "--checkpoint-every", "1",
        "--snapshot-every", "0",
        "--device", "cpu",
    ]
    subprocess.run(common + ["--steps", "2", "--out", str(run_a)], check=True)

    run_b = tmp_path / "phase_b"
    subprocess.run(
        common + ["--steps", "2", "--out", str(run_b),
                  "--resume-from", str(run_a / "last.safetensors")],
        check=True,
    )
    # Phase B writes a config recording the resume_from arg.
    cfg = json.loads((run_b / "config.json").read_text())
    assert cfg["args"]["resume_from"] == str(run_a / "last.safetensors")
    # Phase B last.safetensors should differ from phase A last.safetensors
    # because training continues (phase B took two more steps).
    a = (run_a / "last.safetensors").read_bytes()
    b = (run_b / "last.safetensors").read_bytes()
    assert a != b


def test_trainer_resume_rejects_non_v6_checkpoint(tiny_dataset, tmp_path):
    """A non-V6 (or empty) checkpoint must fail the --resume-from version check."""
    bogus = tmp_path / "bogus.safetensors"
    from safetensors.torch import save_file
    import torch as _torch
    # A fake checkpoint with the wrong version buffer.
    save_file({"slotnet_version": _torch.tensor([50], dtype=_torch.int32)}, str(bogus))
    run_dir = tmp_path / "phase_b"
    result = subprocess.run(
        [
            sys.executable, str(TRAINER),
            "--train", str(tiny_dataset / "train.csv"),
            "--skin-source", str(ROOT / "assets/default_skin"),
            "--steps", "1",
            "--batch", "1",
            "--base-channels", "8",
            "--style-dim", "32",
            "--head-channels", "16",
            "--attn-dim", "32",
            "--attention-heads", "4",
            "--cross-attention-layers", "1",
            "--file-embedding-dim", "8",
            "--query-grid-divisor", "4",
            "--num-workers", "0",
            "--checkpoint-every", "1",
            "--snapshot-every", "0",
            "--out", str(run_dir),
            "--device", "cpu",
            "--resume-from", str(bogus),
        ],
        capture_output=True,
    )
    assert result.returncode != 0
    assert b"V6 (60)" in result.stderr or b"V6 (60)" in result.stdout


def test_eval_runs_and_reports_acceptance_keys(tiny_dataset, tmp_path):
    # First train a tiny model so a checkpoint exists.
    run_dir = tmp_path / "v6_run"
    subprocess.run(
        [
            sys.executable, str(TRAINER),
            "--train", str(tiny_dataset / "train.csv"),
            "--skin-source", str(ROOT / "assets/default_skin"),
            "--steps", "2",
            "--batch", "1",
            "--base-channels", "8",
            "--style-dim", "32",
            "--head-channels", "16",
            "--attn-dim", "32",
            "--attention-heads", "4",
            "--cross-attention-layers", "1",
            "--file-embedding-dim", "8",
            "--query-grid-divisor", "4",
            "--num-workers", "0",
            "--checkpoint-every", "1",
            "--snapshot-every", "0",
            "--out", str(run_dir),
            "--device", "cpu",
        ],
        check=True,
    )
    eval_json = run_dir / "eval.json"
    subprocess.run(
        [
            sys.executable, str(EVAL_SCRIPT),
            "--samples", str(tiny_dataset / "train.csv"),
            "--skin-source", str(ROOT / "assets/default_skin"),
            "--slotnet", str(run_dir / "last.safetensors"),
            "--batch", "1",
            "--device", "cpu",
            "--out-json", str(eval_json),
        ],
        check=True,
    )
    assert eval_json.exists()
    result = json.loads(eval_json.read_text())
    agg = result["aggregate"]
    assert {"visible_mae", "uv_median_px", "copy_conf_auc"} <= set(agg)
    # All 11 trainable files present.
    from atlas_ai.export_spec import TRAINABLE_EXPORT_SPECS
    assert set(result["per_file"]) == {spec.file_name for spec in TRAINABLE_EXPORT_SPECS}
    for spec in TRAINABLE_EXPORT_SPECS:
        keys = result["per_file"][spec.file_name]
        assert {"visible_mae", "uv_median_px", "copy_conf_auc", "visible_pixels"} <= set(keys)
