#!/usr/bin/env python
"""Render 3D body figure for results/best_val_example.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Render skeleton body figure for best val example.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    bench_root = Path(__file__).resolve().parents[1]
    if str(bench_root) not in sys.path:
        sys.path.insert(0, str(bench_root))

    from src.body_visualization import load_best_example_arrays, render_activation_skeleton_montage
    from src.loaders import _resolve_path, load_config

    cfg = load_config(args.config)
    results_dir = _resolve_path(cfg["paths"]["results_dir"])
    json_path = args.json or (results_dir / "best_val_example.json")
    meta = json.loads(json_path.read_text(encoding="utf-8"))

    train_cfg_path = _resolve_path(cfg["paths"]["musclemap_train_config"])
    train_cfg = yaml.safe_load(train_cfg_path.read_text(encoding="utf-8"))
    smplx_win, pred, muscle_names, seq_dir = load_best_example_arrays(meta, cfg, train_cfg)
    meta["sequence_dir"] = str(seq_dir)

    out_png = args.output or (results_dir / "plots" / "best_val_example_body.png")
    title = (
        f"{meta['prompt']}\n"
        f"{meta['checkpoint']} · MAE={float(meta['mae']):.4f}"
    )
    render_activation_skeleton_montage(
        smplx_win,
        pred,
        muscle_names,
        out_png,
        title=title,
    )
    meta["body_visualization_png"] = str(out_png.resolve())
    json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {out_png}")
    print(f"Updated {json_path}")


if __name__ == "__main__":
    main()
