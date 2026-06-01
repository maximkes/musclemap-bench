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
    p.add_argument(
        "--export-timings",
        action="store_true",
        help="Write results/per_sample_timings.json with per-sequence inference times.",
    )
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
        load_musclemap,
        load_test_dataset,
    )
    from src.paired_eval import aggregate_paired_layer1, build_paired_mapping
    from src.align import align_lengths
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
    mapping = build_paired_mapping(cfg, raj_names)

    mm_samples: list[dict] = []
    kin_samples: list[dict] = []
    mm_paired_samples: list[dict] = []
    mm_times_paired: list[float] = []
    kin_times_paired: list[float] = []
    mg_times: list[float] = []
    per_sample_timings: dict[str, list[dict]] = {"musclemap": [], "kinesis": []}

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
            pred, ref_a = align_lengths(mm.activations, ref)
            mm_samples.append(compute_l1_metrics(
                pred,
                ref_a,
                raj_names,
                compute_coactivation=cfg["layer1"]["compute_coactivation"],
                **l1_kw,
            ))

            if mapping is not None and seq_id in k_manifest:
                mm_times_paired.append(float(mm.timing_s))
                if args.export_timings:
                    per_sample_timings["musclemap"].append(
                        {
                            "sequence_id": seq_id,
                            "timing_s": float(mm.timing_s),
                            "T": int(ref.shape[0]),
                            "n_muscles": int(ref.shape[1]),
                        }
                    )
                mm_pred_p = pred[:, mapping.rajagopal_indices]
                mm_ref_p = ref_a[:, mapping.rajagopal_indices]
                mm_paired_samples.append(compute_l1_metrics(
                    mm_pred_p,
                    mm_ref_p,
                    mapping.shared_names,
                    **paired_l1_kw,
                ))

                kin = run_kinesis_from_artifact(k_manifest[seq_id]["path"], seq_id, text)
                kin_timing = float(k_manifest[seq_id].get("timing_s", float("nan")))
                kin_times_paired.append(kin_timing)
                if args.export_timings:
                    per_sample_timings["kinesis"].append(
                        {
                            "sequence_id": seq_id,
                            "timing_s": kin_timing,
                            "T": int(kin.activations.shape[0]) if kin.activations is not None else None,
                            "n_muscles": int(kin.activations.shape[1]) if kin.activations is not None else None,
                        }
                    )
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
    if mapping is not None and mm_paired_samples:
        layer1_paired = aggregate_paired_layer1(mm_paired_samples, kin_samples, mapping)

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
        "resources": compute_resource_summary(mm_times_paired, kin_times_paired, mg_times, cfg),
    }

    out = results_dir / "results.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    report.generate_all(results, results_dir)
    print(f"[bench] wrote {out}")

    if args.export_timings:
        timings_path = results_dir / "per_sample_timings.json"
        timings_path.write_text(json.dumps(per_sample_timings, indent=2), encoding="utf-8")
        print(f"[bench] wrote {timings_path}")


if __name__ == "__main__":
    main()
