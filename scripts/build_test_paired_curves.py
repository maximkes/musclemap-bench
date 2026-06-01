#!/usr/bin/env python
"""Evaluate all checkpoints on test-paired windows and write test_paired_metrics.csv."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build test-paired metrics curve over checkpoints.")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--device", default=None)
    p.add_argument(
        "--epochs",
        nargs="*",
        type=int,
        default=None,
        help="Only evaluate these epoch numbers (default: all epoch_*.pt).",
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Evaluate a single checkpoint path instead of scanning checkpoints_dir.",
    )
    p.add_argument(
        "--append",
        action="store_true",
        help="Append/replace rows in existing CSV instead of overwriting.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    bench_root = Path(__file__).resolve().parents[1]
    if str(bench_root) not in sys.path:
        sys.path.insert(0, str(bench_root))

    from src.curve_data import attach_global_steps, checkpoint_stem, discover_checkpoints, parse_epoch
    from src.loaders import _resolve_path, load_config, load_kinesis_manifest, load_musclemap, load_test_dataset
    from src.metrics_l1 import aggregate_l1_metrics
    from src.paired_eval import (
        build_paired_mapping,
        evaluate_paired_musclemap,
        list_paired_indices,
        paired_coverage_stats,
    )

    cfg = load_config(args.config)
    if args.device is not None:
        cfg["inference"]["device"] = args.device
    device = cfg["inference"]["device"]

    train_cfg = yaml.safe_load(
        _resolve_path(cfg["paths"]["musclemap_train_config"]).read_text(encoding="utf-8")
    )
    curve_cfg = cfg.get("training_curve", {})

    print("[paired-curves] loading test dataset")
    dataset, raj_names = load_test_dataset(cfg)
    k_manifest = load_kinesis_manifest(cfg)
    stats = paired_coverage_stats(dataset, k_manifest)
    print(
        "[paired-curves] coverage: "
        f"{stats['n_paired_windows']}/{stats['n_test_windows']} windows, "
        f"{stats['n_paired_sequences']}/{stats['n_test_sequences']} sequences"
    )

    mapping = build_paired_mapping(cfg, raj_names)
    if mapping is None or len(mapping) == 0:
        print("[paired-curves] muscle mapping unavailable — run Kinesis precompute first")
        sys.exit(1)

    paired_indices = list_paired_indices(dataset, k_manifest)
    if not paired_indices:
        print("[paired-curves] no paired windows found")
        sys.exit(1)

    if args.checkpoint:
        checkpoints = [_resolve_path(args.checkpoint)]
    else:
        ckpt_dir = _resolve_path(cfg["paths"]["checkpoints_dir"])
        checkpoints = discover_checkpoints(ckpt_dir)
        if args.epochs:
            wanted = {int(e) for e in args.epochs}
            checkpoints = [p for p in checkpoints if parse_epoch(p) in wanted]

    if not checkpoints:
        print("[paired-curves] no checkpoints to evaluate")
        sys.exit(1)

    curves_dir = _resolve_path(cfg["paths"]["curves_dir"])
    curves_dir.mkdir(parents=True, exist_ok=True)
    csv_path = curves_dir / "test_paired_metrics.csv"

    existing = pd.DataFrame()
    if args.append and csv_path.exists():
        existing = pd.read_csv(csv_path)

    rows: list[dict] = []
    model = None
    for ckpt in checkpoints:
        stem = checkpoint_stem(ckpt)
        epoch = parse_epoch(ckpt)
        print(f"[paired-curves] epoch {epoch:04d} ({stem}) on {device}")
        if model is None:
            model = load_musclemap(cfg, device=device, checkpoint_path=ckpt)
        else:
            import torch

            state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
            sd = state.get("model", state)
            model.load_state_dict(sd, strict=False)
            model.to(device)
            model.eval()
        mm_samples, _, _ = evaluate_paired_musclemap(
            model,
            dataset,
            paired_indices,
            mapping,
            cfg,
            device=device,
            collect_timings=False,
        )
        agg = aggregate_l1_metrics(mm_samples)
        rows.append(
            {
                "stem": stem,
                "epoch": epoch,
                "mae": float(agg["mae"]),
                "rmse": float(agg["rmse"]),
                "n_paired_windows": len(mm_samples),
            }
        )

    df = pd.DataFrame(rows)
    df = attach_global_steps(df, train_cfg, curve_cfg=curve_cfg)
    df = df[["stem", "epoch", "global_step", "mae", "rmse", "n_paired_windows"]]

    if not existing.empty:
        df = pd.concat([existing[~existing["stem"].isin(df["stem"])], df], ignore_index=True)
        df = df.sort_values("epoch").reset_index(drop=True)

    df.to_csv(csv_path, index=False)
    best = df.loc[df["mae"].idxmin()]
    print(f"[paired-curves] wrote {csv_path} ({len(df)} rows)")
    print(
        f"[paired-curves] best paired MAE: epoch {int(best['epoch'])} "
        f"MAE={best['mae']:.4f} RMSE={best['rmse']:.4f} "
        f"(n={int(best['n_paired_windows'])})"
    )


if __name__ == "__main__":
    main()
