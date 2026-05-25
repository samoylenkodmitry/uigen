"""Smoke tests for V7 completer trainer + eval."""

from __future__ import annotations

import json
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import torch

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
        assert {"step", "epoch", "total", "l1", "hidden_l1", "full_l1",
                "obs_passthrough", "file_name"} <= set(entry)


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
    hidden_keys = {"hidden_supported_mae", "hidden_hit5", "hidden_sobel_mae",
                   "observed_passthrough_mae"}
    agg = result["aggregate"]
    assert {"supported_mae", "hit5", "sobel_mae", "mask_samples"} <= set(agg)
    assert hidden_keys <= set(agg)
    assert agg["mask_samples"] == 2
    assert set(result["per_file"]) == set(TRAINABLE_EXPORT_FILES)
    for file_name in TRAINABLE_EXPORT_FILES:
        entry = result["per_file"][file_name]
        assert {"supported_mae", "hit5", "sobel_mae", "samples"} <= set(entry)
        assert hidden_keys <= set(entry)
    # Per-skin breakdown: with one skin id ("default"), there should be
    # exactly one entry and its shape mirrors the per-file shape.
    assert set(result["per_skin"]) == {"default"}
    skin_entry = result["per_skin"]["default"]
    assert {"supported_mae", "hit5", "sobel_mae", "samples"} <= set(skin_entry)
    assert skin_entry["samples"] > 0


def test_eval_per_skin_metrics_are_not_batch_averages():
    """Same-file batches can contain multiple skins. Per-skin metrics must
    use each sample's own prediction, not the batch mean assigned to both."""
    spec = importlib.util.spec_from_file_location("eval_v7", EVAL_SCRIPT)
    assert spec and spec.loader
    eval_v7 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_v7)

    class _Dataset:
        def set_epoch(self, _epoch): pass

    class _Loader:
        dataset = _Dataset()
        def __iter__(self):
            h, w = 4, 4
            yield {
                "file_name": ["MAIN.bmp", "MAIN.bmp"],
                "skin_id": ["good", "bad"],
                "observed_rgb": torch.stack([
                    torch.zeros(3, h, w),
                    torch.ones(3, h, w),
                ]),
                "observed_mask": torch.ones(2, 1, h, w),
                "target_rgb": torch.zeros(2, 3, h, w),
            }

    class _IdentityModel:
        def eval(self): pass
        def __call__(self, observed_rgb, _observed_mask, _file_id):
            return observed_rgb

    result = eval_v7.evaluate(
        _IdentityModel(),
        _Loader(),
        {"MAIN.bmp": torch.ones(4, 4, dtype=torch.bool)},
        torch.device("cpu"),
        mask_samples=1,
    )
    assert result["per_file"]["MAIN.bmp"]["supported_mae"] == pytest.approx(0.5)
    assert result["per_skin"]["good"]["supported_mae"] == pytest.approx(0.0)
    assert result["per_skin"]["good"]["hit5"] == pytest.approx(1.0)
    assert result["per_skin"]["bad"]["supported_mae"] == pytest.approx(1.0)
    assert result["per_skin"]["bad"]["hit5"] == pytest.approx(0.0)

def _load_eval_module():
    spec = importlib.util.spec_from_file_location("eval_v7", EVAL_SCRIPT)
    assert spec and spec.loader
    eval_v7 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_v7)
    return eval_v7


def test_eval_state_family_only_skips_non_alternative_files():
    """A pure state_family eval must not crash on files with no alternatives.
    It skips them, reports coverage, and still emits per_mode hidden metrics."""
    from atlas_ai.dataset_v7_completion import V7CompletionDataset
    from atlas_ai.support_mask import load_support_masks
    from atlas_ai.v7_masks import V7MaskWeights
    from torch.utils.data import DataLoader

    eval_v7 = _load_eval_module()
    ds = V7CompletionDataset(
        skin_sources={"default": str(DEFAULT_SKIN)},
        state_families_path=str(CONFIG),
        mask_weights=V7MaskWeights(provenance=0.0, state_family=1.0,
                                   random_rect=0.0, whole_file=0.0, passthrough=0.0),
        seed=0,
    )
    eligible, skipped = eval_v7.eval_file_coverage(ds)
    # Component-only files have no alternatives -> skipped under state_family-only.
    assert {"POSBAR.bmp", "MAIN.bmp", "TITLEBAR.bmp", "PLEDIT.bmp"} <= set(skipped)
    assert {"VOLUME.bmp", "BALANCE.bmp", "CBUTTONS.bmp", "EQMAIN.bmp"} <= set(eligible)
    # Sampling a skipped file directly is exactly the crash we avoid by skipping.
    posbar_idx = next(i for i, (_s, f) in enumerate(ds.items) if f == "POSBAR.bmp")
    with pytest.raises(ValueError):
        ds[posbar_idx]

    # End-to-end evaluate over the eligible files only: no crash, with coverage
    # and per_mode hidden metrics in the result.
    sampler = eval_v7.SameFileBatchSampler(
        ds.items, batch_size=2, shuffle=False, include_files=set(eligible),
    )
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=0)

    class _Identity:
        def eval(self): pass
        def __call__(self, observed_rgb, _mask, _file_id):
            return observed_rgb

    result = eval_v7.evaluate(_Identity(), loader, load_support_masks(),
                              torch.device("cpu"), mask_samples=1)
    assert {"POSBAR.bmp", "MAIN.bmp"} <= set(result["coverage"]["skipped_files"])
    assert "POSBAR.bmp" not in result["per_file"]
    assert "VOLUME.bmp" in result["per_file"]
    # A state_family-only mix yields exactly one mode bucket.
    assert set(result["per_mode"]) == {"state_family"}
    assert {"hidden_supported_mae", "hidden_hit5", "hidden_sobel_mae",
            "observed_passthrough_mae"} <= set(result["per_mode"]["state_family"])


def test_eval_coverage_empty_when_no_mode_available():
    """All-zero mask weights leave no eligible file; main() turns this into a
    clear 'not a valid eval mix' error rather than doing work."""
    from atlas_ai.dataset_v7_completion import V7CompletionDataset
    from atlas_ai.v7_masks import V7MaskWeights

    eval_v7 = _load_eval_module()
    ds = V7CompletionDataset(
        skin_sources={"default": str(DEFAULT_SKIN)},
        state_families_path=str(CONFIG),
        mask_weights=V7MaskWeights(provenance=0.0, state_family=0.0,
                                   random_rect=0.0, whole_file=0.0, passthrough=0.0),
        seed=0,
    )
    eligible, skipped = eval_v7.eval_file_coverage(ds)
    assert eligible == []
    assert set(skipped) == set(TRAINABLE_EXPORT_FILES)


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


def test_trainer_with_skin_embedding_two_skins(tmp_path):
    """Trainer + eval round-trip with --skin-embedding-dim and two skin sources.

    Verifies: trainer builds with num_skins=2, eval auto-detects num_skins from
    the checkpoint, per-skin eval entries appear for both ids.
    """
    import shutil
    sib_a = tmp_path / "skin_a"
    sib_b = tmp_path / "skin_b"
    shutil.copytree(DEFAULT_SKIN, sib_a)
    shutil.copytree(DEFAULT_SKIN, sib_b)
    run = tmp_path / "v7_run_skin"
    subprocess.run([
        sys.executable, str(TRAINER),
        "--state-families", str(CONFIG),
        "--skin-sources", f"alpha={sib_a},beta={sib_b}",
        "--batch", "2",
        "--base-channels", "8",
        "--file-embedding-dim", "8",
        "--skin-embedding-dim", "4",
        "--num-workers", "0",
        "--checkpoint-every", "1",
        "--device", "cpu",
        "--out", str(run),
        "--steps", "3",
        "--seed", "11",
    ], check=True)
    config = json.loads((run / "config.json").read_text())
    assert config["num_skins"] == 2
    assert config["skin_embedding_dim"] == 4
    assert config["skin_id_to_index"] == {"alpha": 0, "beta": 1}
    assert config["model_version"] == 71

    eval_json = run / "eval.json"
    subprocess.run([
        sys.executable, str(EVAL_SCRIPT),
        "--state-families", str(CONFIG),
        "--skin-sources", f"alpha={sib_a},beta={sib_b}",
        "--checkpoint", str(run / "last.safetensors"),
        "--batch", "2",
        "--mask-samples", "1",
        "--device", "cpu",
        "--out-json", str(eval_json),
    ], check=True)
    result = json.loads(eval_json.read_text())
    assert set(result["per_skin"]) == {"alpha", "beta"}


def test_skin_resume_rejects_mismatched_num_skins(tmp_path):
    """--resume-from must reject checkpoints whose num_skins disagrees with
    the new run's dataset size."""
    import shutil
    sib_a = tmp_path / "skin_a"
    sib_b = tmp_path / "skin_b"
    shutil.copytree(DEFAULT_SKIN, sib_a)
    shutil.copytree(DEFAULT_SKIN, sib_b)
    # First run: one skin.
    run1 = tmp_path / "v7_one"
    subprocess.run([
        sys.executable, str(TRAINER),
        "--state-families", str(CONFIG),
        "--skin-sources", f"alpha={sib_a}",
        "--batch", "1",
        "--base-channels", "8",
        "--file-embedding-dim", "8",
        "--skin-embedding-dim", "4",
        "--num-workers", "0",
        "--checkpoint-every", "1",
        "--device", "cpu",
        "--out", str(run1),
        "--steps", "1",
    ], check=True)
    # Second run: two skins, resuming from the one-skin checkpoint -> reject.
    run2 = tmp_path / "v7_two_resume"
    res = subprocess.run([
        sys.executable, str(TRAINER),
        "--state-families", str(CONFIG),
        "--skin-sources", f"alpha={sib_a},beta={sib_b}",
        "--batch", "1",
        "--base-channels", "8",
        "--file-embedding-dim", "8",
        "--skin-embedding-dim", "4",
        "--num-workers", "0",
        "--checkpoint-every", "1",
        "--device", "cpu",
        "--out", str(run2),
        "--steps", "1",
        "--resume-from", str(run1 / "last.safetensors"),
    ], capture_output=True)
    assert res.returncode != 0
    blob = res.stderr + res.stdout
    assert b"num_skins" in blob


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
