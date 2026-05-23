#!/usr/bin/env python3
"""V7+ cloud-portable experiment runner.

Glues a runtime config (`configs/runtime/<env>.yaml`) and an experiment YAML
(`experiments/*.yaml`) together, resolves paths, and invokes the requested
trainer. Writes a manifest into the run directory recording everything
that fed the run.

Path resolution rules for strings inside an experiment YAML:
    @data/... -> runtime.paths.data_dir / ...
    @runs/... -> runtime.paths.runs_dir / ...
    @cache/.. -> runtime.paths.cache_dir / ...
    absolute  -> kept as-is
    else      -> runtime.paths.repo_dir / ...

The experiment file declares:
    name        unique experiment id (used for run dir)
    trainer     repo-relative path to a Python trainer
    out         where the trainer writes; `@runs/...` resolves under runs_dir
    args        mapping[CLI flag without leading --] -> value
                  - true bool turns into a flag with no value
                  - false bool is omitted
                  - lists become comma-joined strings (caller can pre-join too)
    data:       optional sub-mapping that builds skin_sources for V7

Override on the CLI:
    --override foo.bar=value      sets the value at a dotted path in the
                                  experiment dict (e.g. args.steps=200).
    --dry-run                     print the assembled command, do nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.runtime_config import RuntimeConfig, load_runtime_from_env_or_arg


PATH_PREFIXES = {
    "@data/": "data_dir",
    "@runs/": "runs_dir",
    "@cache/": "cache_dir",
    "@repo/": "repo_dir",
}


def _resolve_path_token(token: str, runtime: RuntimeConfig) -> Path:
    """Resolve a path token using @data/ / @runs/ / @cache/ / @repo/ prefixes
    or treat as repo-relative if no prefix and not absolute."""
    for prefix, root in PATH_PREFIXES.items():
        if token.startswith(prefix):
            rel = token[len(prefix):]
            return runtime.resolve(rel, root=root)
    p = Path(token)
    if p.is_absolute():
        return p
    return runtime.resolve(token, root="repo_dir")


def _looks_like_path(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    if any(value.startswith(p) for p in PATH_PREFIXES):
        return True
    return False


def _resolve_value(value: Any, runtime: RuntimeConfig) -> Any:
    """Resolve only @-prefixed path tokens; pass other values through verbatim."""
    if _looks_like_path(value):
        return str(_resolve_path_token(value, runtime))
    return value


def _apply_override(experiment: dict, dotted_key: str, value: str) -> None:
    keys = dotted_key.split(".")
    node = experiment
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            node[k] = {}
        node = node[k]
    # Crude type coercion: ints, floats, bools, else string.
    coerced: Any
    if value.lower() in ("true", "false"):
        coerced = value.lower() == "true"
    else:
        try:
            coerced = int(value)
        except ValueError:
            try:
                coerced = float(value)
            except ValueError:
                coerced = value
    node[keys[-1]] = coerced


def _format_skin_sources(value: Any, runtime: RuntimeConfig) -> str:
    """Build a --skin-sources string from various YAML shapes.

    Accepts:
        str  - resolved via path tokens; if it points at a directory, expand
               into "<basename>=<absolute>" comma list of immediate subdirs.
        dict[skin_id -> path] - build "skin_id=path" comma list.
        list[str] - each entry may be 'skin_id=path' or a bare path.
    """
    if isinstance(value, dict):
        items = []
        for sid, p in value.items():
            absolute = _resolve_path_token(str(p), runtime) if _looks_like_path(str(p)) else Path(p)
            items.append(f"{sid}={absolute}")
        return ",".join(items)
    if isinstance(value, list):
        items = []
        for entry in value:
            entry_str = str(entry)
            if "=" in entry_str:
                sid, p = entry_str.split("=", 1)
                absolute = _resolve_path_token(p, runtime) if _looks_like_path(p) else Path(p)
                items.append(f"{sid}={absolute}")
            else:
                abs_p = _resolve_path_token(entry_str, runtime) if _looks_like_path(entry_str) else Path(entry_str)
                items.append(f"{abs_p.name}={abs_p}")
        return ",".join(items)
    text = str(value)
    p = _resolve_path_token(text, runtime) if _looks_like_path(text) else Path(text)
    if p.is_dir():
        children = sorted(c for c in p.iterdir() if c.is_dir())
        if not children:
            # Treat as a single skin dir if no subdirs.
            return f"{p.name}={p}"
        return ",".join(f"{c.name}={c}" for c in children)
    return f"{p.name}={p}"


def build_command(
    experiment: dict,
    runtime: RuntimeConfig,
) -> tuple[list[str], dict]:
    """Translate the experiment dict into a trainer CLI invocation.

    Returns (argv, manifest) where manifest is a JSON-serializable dict
    capturing the resolved invocation, runtime, and experiment.
    """
    trainer_rel = str(experiment.get("trainer", "")).strip()
    if not trainer_rel:
        raise ValueError("experiment missing 'trainer'")
    trainer_path = runtime.resolve(trainer_rel, root="repo_dir")
    if not trainer_path.exists():
        raise FileNotFoundError(f"trainer not found: {trainer_path}")

    argv: list[str] = [sys.executable, str(trainer_path)]

    args_mapping = experiment.get("args") or {}
    if not isinstance(args_mapping, dict):
        raise ValueError("'args' must be a mapping")

    args_mapping = dict(args_mapping)

    # Synthesize derived args.
    if "skin-sources" not in args_mapping and experiment.get("data"):
        data = experiment["data"]
        if "skin_sources" in data:
            args_mapping["skin-sources"] = _format_skin_sources(data["skin_sources"], runtime)

    # Always wire `--out` from experiment.out if present.
    out_value = experiment.get("out")
    if out_value:
        args_mapping["out"] = _resolve_value(out_value, runtime)

    # Runtime defaults if the experiment doesn't override.
    args_mapping.setdefault("device", runtime.device)
    if "num-workers" not in args_mapping and "num_workers" not in args_mapping:
        args_mapping["num-workers"] = runtime.num_workers
    if runtime.amp and "amp" not in args_mapping:
        args_mapping["amp"] = True

    # Translate to argv.
    for raw_key, raw_value in args_mapping.items():
        flag = "--" + str(raw_key).replace("_", "-")
        if isinstance(raw_value, bool):
            if raw_value:
                argv.append(flag)
            continue
        if raw_value is None:
            continue
        resolved = _resolve_value(raw_value, runtime)
        argv.extend([flag, str(resolved)])

    manifest = {
        "runtime": {
            "name": runtime.name,
            "paths": {k: str(v) for k, v in runtime.paths.items()},
            "num_workers": runtime.num_workers,
            "device": runtime.device,
            "amp": runtime.amp,
        },
        "experiment": experiment,
        "argv": argv,
        "trainer": str(trainer_path),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return argv, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", default=None,
                        help="Path to a runtime YAML; falls back to UIGEN_RUNTIME env var.")
    parser.add_argument("--experiment", required=True,
                        help="Path to an experiment YAML.")
    parser.add_argument("--override", action="append", default=[],
                        help="Dotted-path override, e.g. args.steps=200. May be repeated.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the resolved command and exit without invoking it.")
    args = parser.parse_args()

    runtime = load_runtime_from_env_or_arg(args.runtime)
    experiment_path = Path(args.experiment).resolve()
    if not experiment_path.exists():
        raise SystemExit(f"experiment YAML not found: {experiment_path}")
    experiment = yaml.safe_load(experiment_path.read_text(encoding="utf-8")) or {}
    if not isinstance(experiment, dict):
        raise SystemExit(f"{experiment_path}: top-level must be a mapping")

    for override in args.override:
        if "=" not in override:
            raise SystemExit(f"--override expects key=value, got {override!r}")
        key, value = override.split("=", 1)
        _apply_override(experiment, key, value)

    argv, manifest = build_command(experiment, runtime)

    if args.dry_run:
        print(shlex.join(argv))
        return 0

    out_dir = Path(manifest["argv"][manifest["argv"].index("--out") + 1])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    return subprocess.run(argv).returncode


if __name__ == "__main__":
    raise SystemExit(main())
