#!/usr/bin/env python
"""Measure wall-clock inference on test-paired windows only."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile paired-window inference latency.")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--device", default=None)
    p.add_argument(
        "--checkpoint",
        default=None,
        help="MuscleMAP checkpoint (default: config paths.musclemap_checkpoint).",
    )
    p.add_argument(
        "--out",
        default=None,
        help="JSON output path (default: merge into results/inference_profile.json).",
    )
    return p.parse_args()


def _stats(ts: list[float]) -> dict[str, float | None]:
    import numpy as np

    arr = [float(x) for x in ts if x == x]
    if not arr:
        return {"mean_s": None, "std_s": None, "median_s": None, "n": 0}
    return {
        "mean_s": float(np.mean(arr)),
        "std_s": float(np.std(arr)),
        "median_s": float(np.median(arr)),
        "n": len(arr),
    }


def main() -> None:
    args = parse_args()
    bench_root = Path(__file__).resolve().parents[1]
    if str(bench_root) not in sys.path:
        sys.path.insert(0, str(bench_root))

    from src.loaders import _resolve_path, load_config, load_kinesis_manifest, load_musclemap, load_test_dataset
    from src.paired_eval import (
        build_paired_mapping,
        evaluate_paired_kinesis,
        evaluate_paired_musclemap,
        list_paired_indices,
        paired_coverage_stats,
    )

    cfg = load_config(args.config)
    if args.device is not None:
        cfg["inference"]["device"] = args.device
    device = cfg["inference"]["device"]

    print("[profile-paired] loading test dataset")
    dataset, raj_names = load_test_dataset(cfg)
    k_manifest = load_kinesis_manifest(cfg)
    stats = paired_coverage_stats(dataset, k_manifest)
    print(f"[profile-paired] {stats['n_paired_windows']} paired windows")

    mapping = build_paired_mapping(cfg, raj_names)
    if mapping is None or len(mapping) == 0:
        print("[profile-paired] muscle mapping unavailable")
        sys.exit(1)

    paired_indices = list_paired_indices(dataset, k_manifest)
    ckpt = _resolve_path(args.checkpoint) if args.checkpoint else _resolve_path(cfg["paths"]["musclemap_checkpoint"])

    print(f"[profile-paired] MuscleMAP checkpoint {ckpt.name} on {device}")
    model = load_musclemap(cfg, device=device, checkpoint_path=ckpt)
    _, mm_times, _ = evaluate_paired_musclemap(
        model,
        dataset,
        paired_indices,
        mapping,
        cfg,
        device=device,
        collect_timings=True,
    )
    _, kin_times = evaluate_paired_kinesis(
        dataset,
        paired_indices,
        mapping,
        cfg,
        collect_timings=True,
    )

    paired_timing: dict[str, Any] = {
        "n_paired_windows": stats["n_paired_windows"],
        "n_paired_sequences": stats["n_paired_sequences"],
        "checkpoint": ckpt.name,
        "device": device,
        "musclemap": _stats(mm_times),
        "kinesis": {
            **_stats(kin_times),
            "note": "Kinesis timings from precompute manifest (MyoLeg simulation wall-clock).",
        },
    }

    results_dir = _resolve_path(cfg["paths"]["results_dir"])
    out_path = Path(args.out) if args.out else results_dir / "inference_profile.json"
    profile: dict[str, Any] = {}
    if out_path.exists():
        profile = json.loads(out_path.read_text(encoding="utf-8"))
    profile["paired_inference"] = paired_timing
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")

    print(f"[profile-paired] MuscleMAP mean={paired_timing['musclemap']['mean_s']:.4f}s")
    print(f"[profile-paired] Kinesis mean={paired_timing['kinesis']['mean_s']:.4f}s")
    print(f"[profile-paired] wrote {out_path}")


if __name__ == "__main__":
    main()
