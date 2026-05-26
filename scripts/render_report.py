#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src import report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/results.json")
    args = parser.parse_args()
    path = Path(args.results)
    data = json.loads(path.read_text(encoding="utf-8"))
    report.generate_all(data, path.parent)


if __name__ == "__main__":
    main()
