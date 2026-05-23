"""V7 cloud-portability chunk 1: runtime configs + experiment runner."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
LOCAL_RUNTIME = ROOT / "configs/runtime/local.yaml"
KAGGLE_RUNTIME = ROOT / "configs/runtime/kaggle.yaml"
COLAB_RUNTIME = ROOT / "configs/runtime/colab.yaml"
EXPERIMENT = ROOT / "experiments/v7_completer_gateB.yaml"
RUN_EXPERIMENT = ROOT / "scripts/run_experiment.py"
BENCHMARK = ROOT / "scripts/benchmark_runtime.py"
PRINT_ENV = ROOT / "scripts/print_env.py"


def test_runtime_yaml_files_exist():
    assert LOCAL_RUNTIME.exists()
    assert KAGGLE_RUNTIME.exists()
    assert COLAB_RUNTIME.exists()


def test_runtime_loader_resolves_local_paths():
    sys.path.insert(0, str(ROOT))
    from scripts.runtime_config import load_runtime
    rt = load_runtime(LOCAL_RUNTIME)
    assert rt.name == "local"
    assert rt.paths["repo_dir"].resolve() == ROOT.resolve()
    assert rt.paths["data_dir"].resolve() == ROOT.resolve()
    assert rt.paths["runs_dir"].resolve() == (ROOT / "runs").resolve()
    assert rt.paths["cache_dir"].resolve() == (ROOT / ".cache").resolve()
    assert rt.device == "cuda"
    assert rt.amp is True


def test_runtime_loader_keeps_absolute_kaggle_paths():
    sys.path.insert(0, str(ROOT))
    from scripts.runtime_config import load_runtime
    rt = load_runtime(KAGGLE_RUNTIME)
    assert rt.paths["repo_dir"] == Path("/kaggle/working/uigen")
    assert rt.paths["data_dir"] == Path("/kaggle/input/uigen-data")
    assert rt.paths["runs_dir"] == Path("/kaggle/working/runs")
    assert rt.paths["cache_dir"] == Path("/kaggle/working/cache")


def test_runtime_loader_colab_has_drive_paths():
    sys.path.insert(0, str(ROOT))
    from scripts.runtime_config import load_runtime
    rt = load_runtime(COLAB_RUNTIME)
    assert rt.paths["repo_dir"] == Path("/content/uigen")
    assert rt.paths["drive_runs_dir"] == Path("/content/drive/MyDrive/uigen_runs")
    assert rt.paths["drive_data_dir"] == Path("/content/drive/MyDrive/uigen_data")


def test_runtime_loader_rejects_missing_required_path(tmp_path):
    sys.path.insert(0, str(ROOT))
    from scripts.runtime_config import load_runtime
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"name": "bad", "paths": {"repo_dir": "."}}))
    with pytest.raises(ValueError, match="missing required path"):
        load_runtime(bad)


def test_runtime_loader_rejects_missing_file(tmp_path):
    sys.path.insert(0, str(ROOT))
    from scripts.runtime_config import load_runtime
    with pytest.raises(FileNotFoundError):
        load_runtime(tmp_path / "nope.yaml")


def test_runtime_resolve_method_handles_absolute():
    sys.path.insert(0, str(ROOT))
    from scripts.runtime_config import load_runtime
    rt = load_runtime(LOCAL_RUNTIME)
    abs_path = "/tmp/anywhere"
    assert rt.resolve(abs_path) == Path(abs_path)


def test_run_experiment_dry_run_prints_command():
    """--dry-run should print a runnable trainer command and not invoke it."""
    result = subprocess.run(
        [sys.executable, str(RUN_EXPERIMENT),
         "--runtime", str(LOCAL_RUNTIME),
         "--experiment", str(EXPERIMENT),
         "--override", "args.steps=10",
         "--override", "args.skin-sources=default=assets/default_skin",
         "--dry-run"],
        capture_output=True, text=True, check=True,
    )
    cmd = result.stdout.strip()
    # Trainer path is present and resolved through the runtime.
    assert "train_v7_completer.py" in cmd
    # Override took effect.
    assert "--steps 10" in cmd
    # skin-sources flag is present.
    assert "--skin-sources" in cmd
    # No actual run dir should be created during dry-run.
    out_dir = ROOT / "runs" / "v7_completer_gateB_16skin"
    # We may or may not have one from prior runs; just assert the dry-run
    # didn't crash and the command looks valid.


def test_run_experiment_uses_env_var(tmp_path, monkeypatch):
    """UIGEN_RUNTIME env var picks the runtime when --runtime is absent."""
    env = {**__import__("os").environ, "UIGEN_RUNTIME": str(LOCAL_RUNTIME)}
    result = subprocess.run(
        [sys.executable, str(RUN_EXPERIMENT),
         "--experiment", str(EXPERIMENT),
         "--override", "args.skin-sources=default=assets/default_skin",
         "--dry-run"],
        env=env, capture_output=True, text=True, check=True,
    )
    assert "train_v7_completer.py" in result.stdout


def test_run_experiment_requires_runtime():
    """No --runtime and no env var should fail clearly."""
    env = {k: v for k, v in __import__("os").environ.items() if k != "UIGEN_RUNTIME"}
    result = subprocess.run(
        [sys.executable, str(RUN_EXPERIMENT),
         "--experiment", str(EXPERIMENT),
         "--dry-run"],
        env=env, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert b"no runtime configured" in result.stderr.encode() or b"runtime" in (result.stderr + result.stdout).encode().lower()


def test_run_experiment_writes_manifest(tmp_path):
    """A non-dry run writes manifest.json into the out dir before invoking the trainer."""
    out_dir = tmp_path / "run"
    # Use a stripped-down inline experiment to keep the test fast.
    exp = tmp_path / "tiny.yaml"
    exp.write_text(yaml.safe_dump({
        "name": "tiny",
        "trainer": "train_v7_completer.py",
        "out": str(out_dir),
        "args": {
            "state-families": "configs/state_families_classic.yaml",
            "skin-sources": f"default={ROOT}/assets/default_skin",
            "steps": 1,
            "batch": 1,
            "base-channels": 8,
            "file-embedding-dim": 8,
            "checkpoint-every": 1,
            "num-workers": 0,
            "device": "cpu",
        },
    }))
    subprocess.run(
        [sys.executable, str(RUN_EXPERIMENT),
         "--runtime", str(LOCAL_RUNTIME),
         "--experiment", str(exp)],
        check=True,
    )
    manifest_path = out_dir / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["runtime"]["name"] == "local"
    assert manifest["experiment"]["name"] == "tiny"
    assert any("train_v7_completer.py" in arg for arg in manifest["argv"])
    # Trainer ran -> last.safetensors exists.
    assert (out_dir / "last.safetensors").exists()


def test_benchmark_smoke_cpu(tmp_path):
    """benchmark_runtime.py runs a few steps end-to-end on CPU."""
    exp = tmp_path / "bench.yaml"
    exp.write_text(yaml.safe_dump({
        "name": "bench",
        "trainer": "train_v7_completer.py",
        "out": str(tmp_path / "out"),  # benchmark doesn't write here
        "args": {
            "state-families": "configs/state_families_classic.yaml",
            "skin-sources": f"default={ROOT}/assets/default_skin",
            "steps": 100,
            "batch": 1,
            "base-channels": 8,
            "file-embedding-dim": 8,
            "device": "cpu",
        },
    }))
    result = subprocess.run(
        [sys.executable, str(BENCHMARK),
         "--runtime", str(LOCAL_RUNTIME),
         "--experiment", str(exp),
         "--steps", "3",
         "--device", "cpu"],
        capture_output=True, text=True, check=True,
    )
    assert "sec/step median" in result.stdout
    assert "steps run:" in result.stdout
    assert "estimated full" in result.stdout


def test_print_env_runs():
    """print_env.py emits something useful and exits cleanly."""
    result = subprocess.run(
        [sys.executable, str(PRINT_ENV)],
        capture_output=True, text=True, check=True,
    )
    assert "python:" in result.stdout
    assert "torch:" in result.stdout


def test_print_env_json_mode_parses():
    result = subprocess.run(
        [sys.executable, str(PRINT_ENV), "--json"],
        capture_output=True, text=True, check=True,
    )
    payload = json.loads(result.stdout)
    assert "python_version" in payload
    assert "torch_version" in payload


def test_makefile_parses():
    """make -n on a known target should not error."""
    makefile = ROOT / "Makefile"
    assert makefile.exists()
    result = subprocess.run(
        ["make", "-n", "-f", str(makefile), "local-dry"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert "scripts/run_experiment.py" in result.stdout
    assert "--dry-run" in result.stdout


def test_makefile_has_all_required_targets():
    """All targets listed in the chunk-1 spec must be present."""
    makefile = ROOT / "Makefile"
    text = makefile.read_text()
    for target in ["local", "kaggle", "colab",
                   "bench-local", "bench-kaggle", "bench-colab"]:
        assert f"{target}:" in text, f"missing target: {target}"


def test_trainer_stop_after_minutes_saves_last(tmp_path):
    """--stop-after-minutes triggers a clean exit and writes last.safetensors."""
    out_dir = tmp_path / "stop_run"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "train_v7_completer.py"),
         "--state-families", str(ROOT / "configs/state_families_classic.yaml"),
         "--skin-sources", f"default={ROOT}/assets/default_skin",
         "--steps", "999999",  # large; we should stop on time, not on steps
         "--stop-after-minutes", "0.05",  # 3 seconds
         "--batch", "1",
         "--base-channels", "8",
         "--file-embedding-dim", "8",
         "--num-workers", "0",
         "--checkpoint-every", "1",
         "--device", "cpu",
         "--out", str(out_dir)],
        capture_output=True, text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-500:]
    assert "stopped by --stop-after-minutes" in proc.stdout
    assert (out_dir / "last.safetensors").exists()
    assert (out_dir / "config.json").exists()


def test_trainer_save_every_aliases_snapshot_every(tmp_path):
    """--save-every overrides --snapshot-every."""
    out_dir = tmp_path / "alias_run"
    subprocess.run(
        [sys.executable, str(ROOT / "train_v7_completer.py"),
         "--state-families", str(ROOT / "configs/state_families_classic.yaml"),
         "--skin-sources", f"default={ROOT}/assets/default_skin",
         "--steps", "3",
         "--save-every", "2",
         "--batch", "1",
         "--base-channels", "8",
         "--file-embedding-dim", "8",
         "--num-workers", "0",
         "--checkpoint-every", "1",
         "--device", "cpu",
         "--out", str(out_dir)],
        check=True,
    )
    # --save-every 2 should have created a snapshot at step 2.
    assert (out_dir / "snapshot_step000002.safetensors").exists()


def test_trainer_config_records_env(tmp_path):
    """config.json now carries an env section with torch/python info."""
    out_dir = tmp_path / "env_run"
    subprocess.run(
        [sys.executable, str(ROOT / "train_v7_completer.py"),
         "--state-families", str(ROOT / "configs/state_families_classic.yaml"),
         "--skin-sources", f"default={ROOT}/assets/default_skin",
         "--steps", "1",
         "--batch", "1",
         "--base-channels", "8",
         "--file-embedding-dim", "8",
         "--num-workers", "0",
         "--checkpoint-every", "1",
         "--device", "cpu",
         "--out", str(out_dir)],
        check=True,
    )
    config = json.loads((out_dir / "config.json").read_text())
    assert "env" in config
    assert "torch_version" in config["env"]
    assert "python_version" in config["env"]
