#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cmd = [
        sys.executable,
        "scripts/run_benchmark.py",
        "--config", args.config,
        "--max-samples", "5",
        "--skip-layer2",
    ]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
