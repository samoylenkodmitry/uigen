"""Runtime config loader for V7+ cloud portability.

Runtime YAMLs live in `configs/runtime/<env>.yaml`. They declare path roots
and worker / device defaults for a target environment (local / Kaggle /
Colab / etc.). Experiment YAMLs reference paths relative to those roots; the
loader resolves them into absolute paths.

Public API:

    load_runtime(path)            -> RuntimeConfig
    resolve_in(runtime, rel, root="repo_dir") -> absolute Path

The runtime config also carries `device`, `amp`, and `num_workers`
defaults the experiment runner uses unless the experiment overrides them.

This module is intentionally narrow: it just loads and resolves paths.
Higher-level orchestration lives in scripts/run_experiment.py.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


REQUIRED_PATHS = ("repo_dir", "data_dir", "runs_dir", "cache_dir")


@dataclass(frozen=True)
class RuntimeConfig:
    """Parsed runtime config.

    `paths` is a mapping[name -> absolute Path]. The required entries
    (repo_dir, data_dir, runs_dir, cache_dir) are always populated.
    Optional entries (drive_runs_dir, drive_data_dir, ...) are exposed
    verbatim and may be missing.

    `name` is the runtime identifier (matches the YAML basename).
    """

    name: str
    paths: dict[str, Path]
    num_workers: int
    device: str
    amp: bool
    raw: dict[str, Any] = field(default_factory=dict)

    def resolve(self, value: str | os.PathLike[str], root: str = "repo_dir") -> Path:
        """Resolve a relative path against the named root. Absolute paths
        are returned unchanged."""
        p = Path(value)
        if p.is_absolute():
            return p
        if root not in self.paths:
            raise KeyError(f"runtime '{self.name}' has no path root {root!r}")
        return self.paths[root] / p

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]


def _resolve_relative_path(value: Any, anchor: Path) -> Path:
    """Resolve a path string from a YAML against an anchor directory.

    Absolute paths stay as-is. Relative paths are joined onto `anchor`
    (typically the directory containing the YAML or the runtime
    repo_dir). The result is NOT resolved to a real file; the caller
    decides whether to require existence.
    """
    p = Path(str(value))
    if p.is_absolute():
        return p
    return (anchor / p)


def load_runtime(path: str | os.PathLike[str]) -> RuntimeConfig:
    """Load and validate a runtime YAML.

    The YAML must declare `name` and a `paths` mapping with the four
    required entries listed in REQUIRED_PATHS. Optional path entries
    (drive_*) are preserved. Relative paths in `paths` are resolved
    relative to the YAML's parent directory; absolute paths are kept.

    `num_workers`, `device`, `amp` default to 0 / "cpu" / False if absent.
    """
    yaml_path = Path(path).resolve()
    if not yaml_path.exists():
        raise FileNotFoundError(f"runtime YAML not found: {yaml_path}")
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{yaml_path}: top-level must be a mapping")
    name = str(data.get("name", yaml_path.stem))
    raw_paths = data.get("paths") or {}
    if not isinstance(raw_paths, dict):
        raise ValueError(f"{yaml_path}: 'paths' must be a mapping")
    for required in REQUIRED_PATHS:
        if required not in raw_paths:
            raise ValueError(f"{yaml_path}: missing required path '{required}'")
    anchor = yaml_path.parent
    paths: dict[str, Path] = {
        key: _resolve_relative_path(value, anchor) for key, value in raw_paths.items()
    }
    num_workers = int(data.get("num_workers", 0))
    device = str(data.get("device", "cpu"))
    amp = bool(data.get("amp", False))
    return RuntimeConfig(
        name=name,
        paths=paths,
        num_workers=num_workers,
        device=device,
        amp=amp,
        raw=data,
    )


def load_runtime_from_env_or_arg(arg: str | None) -> RuntimeConfig:
    """Convenience: explicit arg wins; otherwise UIGEN_RUNTIME env var."""
    path = arg or os.environ.get("UIGEN_RUNTIME")
    if not path:
        raise SystemExit(
            "no runtime configured. Pass --runtime <yaml> or set UIGEN_RUNTIME."
        )
    return load_runtime(path)


__all__ = [
    "RuntimeConfig",
    "REQUIRED_PATHS",
    "load_runtime",
    "load_runtime_from_env_or_arg",
]
