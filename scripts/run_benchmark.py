#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run musclemap-bench")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--skip-layer1", action="store_true")
    p.add_argument("--skip-layer2", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    bench_root = Path(__file__).resolve().parents[1]
    if str(bench_root) not in sys.path:
        sys.path.insert(0, str(bench_root))
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.max_samples is not None:
        cfg["test_set"]["max_samples"] = args.max_samples
    if args.device is not None:
        cfg["inference"]["device"] = args.device

    results_dir = Path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    device = cfg["inference"]["device"]

    from src.loaders import (
        get_benchmark_sample,
        load_kinesis_manifest,
        load_kinesis_muscle_names,
        load_musclemap,
        load_test_dataset,
    )
    from precompute.run_kinesis import RAJAGOPAL_TO_KINESIS_ACTUATOR
    from src.align import build_mapping_kinesis_artifacts, align_lengths
    from src.metrics_l1 import compute_l1_metrics, aggregate_l1_metrics
    from src.resources import compute_resource_summary
    from src import report

    print("[bench] loading MuscleMAP")
    model = load_musclemap(cfg, device=device)
    print("[bench] loading dataset")
    dataset, raj_names = load_test_dataset(cfg)

    indices = list(range(len(dataset)))
    if cfg["test_set"]["max_samples"]:
        rng = np.random.default_rng(cfg["test_set"]["seed"])
        indices = sorted(rng.choice(indices, size=min(cfg["test_set"]["max_samples"], len(indices)), replace=False).tolist())

    k_manifest = load_kinesis_manifest(cfg)
    k_names: list[str] = []
    mapping = None
    try:
        k_names = load_kinesis_muscle_names(cfg)
    except FileNotFoundError:
        pass
    if k_names:
        mapping = build_mapping_kinesis_artifacts(raj_names, k_names, RAJAGOPAL_TO_KINESIS_ACTUATOR)

    mm_samples: list[dict] = []
    kin_samples: list[dict] = []
    mm_paired_samples: list[dict] = []
    mm_times: list[float] = []
    kin_times: list[float] = []
    mg_times: list[float] = []

    l1_kw = dict(
        dtw_enabled=cfg["layer1"]["dtw_enabled"],
        dtw_radius=cfg["layer1"]["dtw_radius"],
        activation_threshold=cfg["layer1"]["activation_threshold"],
        min_active_frames=cfg["layer1"]["min_active_frames"],
    )
    paired_l1_kw = {**l1_kw, "compute_coactivation": False}

    if not args.skip_layer1 and cfg["layer1"]["enabled"]:
        from src.inference import run_musclemap, run_kinesis_from_artifact
        for idx in tqdm(indices, desc="layer1"):
            sample = get_benchmark_sample(dataset, idx)
            seq_id = sample.sequence_id
            text = sample.text
            ref = sample.activations

            mm = run_musclemap(model, text, seq_id, device=device, ref_T=ref.shape[0])
            mm_times.append(mm.timing_s)
            pred, ref_a = align_lengths(mm.activations, ref)
            mm_samples.append(compute_l1_metrics(
                pred,
                ref_a,
                raj_names,
                compute_coactivation=cfg["layer1"]["compute_coactivation"],
                **l1_kw,
            ))

            if mapping is not None and seq_id in k_manifest:
                mm_pred_p = pred[:, mapping.rajagopal_indices]
                mm_ref_p = ref_a[:, mapping.rajagopal_indices]
                mm_paired_samples.append(compute_l1_metrics(
                    mm_pred_p,
                    mm_ref_p,
                    mapping.shared_names,
                    **paired_l1_kw,
                ))

                kin = run_kinesis_from_artifact(k_manifest[seq_id]["path"], seq_id, text)
                kin_times.append(float(k_manifest[seq_id].get("timing_s", float("nan"))))
                kin_pred = kin.activations[:, mapping.kinesis_indices]
                kin_ref = ref[:, mapping.rajagopal_indices]
                kin_pred, kin_ref = align_lengths(kin_pred, kin_ref)
                kin_samples.append(compute_l1_metrics(
                    kin_pred,
                    kin_ref,
                    mapping.shared_names,
                    **paired_l1_kw,
                ))

    layer2_results = {"musclemap": {}, "motiongpt": {}}
    if not args.skip_layer2 and cfg["layer2"]["enabled"]:
        layer2_results = {"musclemap": {}, "motiongpt": {}}

    layer1_paired: dict = {}
    if mm_paired_samples and len(mm_paired_samples) == len(kin_samples):
        layer1_paired = {
            "n_sequences": len(mm_paired_samples),
            "n_muscles": len(mapping.shared_names) if mapping else 0,
            "muscle_names": list(mapping.shared_names) if mapping else [],
            "description": (
                "Paired comparison on sequences with Kinesis precompute artifacts; "
                "both methods scored on the same mapped Rajagopal leg muscles."
            ),
            "musclemap": aggregate_l1_metrics(mm_paired_samples),
            "kinesis": aggregate_l1_metrics(kin_samples),
        }
    elif mm_paired_samples:
        layer1_paired = {
            "error": "paired sample count mismatch",
            "n_musclemap": len(mm_paired_samples),
            "n_kinesis": len(kin_samples),
        }

    results = {
        "meta": {
            "n_samples": len(indices),
            "n_paired_sequences": len(mm_paired_samples),
            "device": device,
        },
        "layer1": {
            "musclemap": aggregate_l1_metrics(mm_samples),
            "kinesis": aggregate_l1_metrics(kin_samples),
        },
        "layer1_paired": layer1_paired,
        "layer2": layer2_results,
        "resources": compute_resource_summary(mm_times, kin_times, mg_times, cfg),
    }

    out = results_dir / "results.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    report.generate_all(results, results_dir)
    print(f"[bench] wrote {out}")


if __name__ == "__main__":
    main()
