#!/usr/bin/env python3
"""Print Python / PyTorch / GPU info for the current environment.

Intended for diagnostics in a new cloud session ("what GPU did I get?"),
and as the env section recorded in run manifests.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys


def collect_env() -> dict:
    env: dict = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch
        env["torch_version"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            try:
                env["torch_cuda_version"] = torch.version.cuda
            except Exception:
                pass
            env["cuda_device_count"] = torch.cuda.device_count()
            devices = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                devices.append({
                    "index": i,
                    "name": props.name,
                    "total_memory_mib": props.total_memory // (1024 * 1024),
                    "multi_processor_count": props.multi_processor_count,
                })
            env["cuda_devices"] = devices
    except ImportError:
        env["torch_version"] = None
        env["cuda_available"] = False
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true",
                        help="Emit a single JSON object instead of human text.")
    args = parser.parse_args()
    env = collect_env()
    if args.json:
        print(json.dumps(env, indent=2, sort_keys=True))
        return 0
    print(f"python:       {env.get('python_version')}")
    print(f"platform:     {env.get('platform')}")
    print(f"torch:        {env.get('torch_version')}")
    print(f"torch cuda:   {env.get('torch_cuda_version', 'n/a')}")
    print(f"cuda avail:   {env.get('cuda_available')}")
    for d in env.get("cuda_devices", []):
        print(f"  GPU {d['index']}: {d['name']} ({d['total_memory_mib']} MiB, "
              f"{d['multi_processor_count']} SMs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
