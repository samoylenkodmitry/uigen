#!/usr/bin/env python3
from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data_v34")
    parser.parse_args()
    raise SystemExit("dataset sharding is reserved until small-file I/O becomes a measured bottleneck")


if __name__ == "__main__":
    main()
