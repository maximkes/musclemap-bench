from __future__ import annotations

from typing import Any

import numpy as np

from precompute.run_kinesis import RAJAGOPAL_TO_KINESIS_ACTUATOR
from src.align import MuscleMapping, align_lengths, build_mapping_kinesis_artifacts
from src.inference import run_kinesis_from_artifact, run_musclemap
from src.loaders import (
    _bench_root,
    _sequence_id_for_index,
    get_benchmark_sample,
    load_kinesis_manifest,
    load_kinesis_muscle_names,
)
from src.metrics_l1 import aggregate_l1_metrics, compute_l1_metrics


def build_paired_mapping(cfg: dict[str, Any], rajagopal_names: list[str]) -> MuscleMapping | None:
    """Build Rajagopal↔Kinesis artifact muscle mapping when names are available."""
    try:
        k_names = load_kinesis_muscle_names(cfg)
    except FileNotFoundError:
        return None
    if not k_names:
        return None
    return build_mapping_kinesis_artifacts(rajagopal_names, k_names, RAJAGOPAL_TO_KINESIS_ACTUATOR)


def list_paired_indices(dataset: Any, k_manifest: dict[str, dict[str, Any]]) -> list[int]:
    """Return test dataset indices whose sequence id has a Kinesis artifact."""
    paired: list[int] = []
    for idx in range(len(dataset)):
        seq_id = _sequence_id_for_index(dataset, idx)
        if seq_id in k_manifest:
            paired.append(idx)
    return paired


def paired_coverage_stats(dataset: Any, k_manifest: dict[str, dict[str, Any]]) -> dict[str, int]:
    """Summarize paired window / unique-sequence coverage on the test split."""
    paired_indices = list_paired_indices(dataset, k_manifest)
    unique_paired: set[str] = set()
    unique_test: set[str] = set()
    for idx in range(len(dataset)):
        seq_id = _sequence_id_for_index(dataset, idx)
        unique_test.add(seq_id)
        if seq_id in k_manifest:
            unique_paired.add(seq_id)
    return {
        "n_test_windows": len(dataset),
        "n_paired_windows": len(paired_indices),
        "n_test_sequences": len(unique_test),
        "n_paired_sequences": len(unique_paired),
        "n_missing_sequences": len(unique_test - unique_paired),
    }


def evaluate_paired_musclemap(
    model: Any,
    dataset: Any,
    paired_indices: list[int],
    mapping: MuscleMapping,
    cfg: dict[str, Any],
    *,
    device: str,
    collect_timings: bool = False,
) -> tuple[list[dict[str, Any]], list[float], list[float]]:
    """Run MuscleMAP on paired test windows and score against GT on mapped leg muscles."""
    layer1 = cfg["layer1"]
    paired_l1_kw = dict(
        dtw_enabled=layer1["dtw_enabled"],
        dtw_radius=layer1["dtw_radius"],
        activation_threshold=layer1["activation_threshold"],
        min_active_frames=layer1["min_active_frames"],
        compute_coactivation=False,
    )
    k_manifest = load_kinesis_manifest(cfg)

    per_sample: list[dict[str, Any]] = []
    mm_times: list[float] = []
    kin_times: list[float] = []

    for idx in paired_indices:
        sample = get_benchmark_sample(dataset, idx)
        seq_id = sample.sequence_id
        ref = sample.activations

        mm = run_musclemap(model, sample.text, seq_id, device=device, ref_T=ref.shape[0])
        if collect_timings:
            mm_times.append(float(mm.timing_s))

        pred, ref_a = align_lengths(mm.activations, ref)
        mm_pred_p = pred[:, mapping.rajagopal_indices]
        mm_ref_p = ref_a[:, mapping.rajagopal_indices]
        per_sample.append(
            compute_l1_metrics(mm_pred_p, mm_ref_p, mapping.shared_names, **paired_l1_kw)
        )

        if collect_timings:
            kin_timing = float(k_manifest[seq_id].get("timing_s", float("nan")))
            kin_times.append(kin_timing)

    return per_sample, mm_times, kin_times


def evaluate_paired_kinesis(
    dataset: Any,
    paired_indices: list[int],
    mapping: MuscleMapping,
    cfg: dict[str, Any],
    *,
    collect_timings: bool = False,
) -> tuple[list[dict[str, Any]], list[float]]:
    """Score precomputed Kinesis artifacts on paired test windows."""
    layer1 = cfg["layer1"]
    paired_l1_kw = dict(
        dtw_enabled=layer1["dtw_enabled"],
        dtw_radius=layer1["dtw_radius"],
        activation_threshold=layer1["activation_threshold"],
        min_active_frames=layer1["min_active_frames"],
        compute_coactivation=False,
    )
    k_manifest = load_kinesis_manifest(cfg)

    per_sample: list[dict[str, Any]] = []
    kin_times: list[float] = []

    for idx in paired_indices:
        sample = get_benchmark_sample(dataset, idx)
        seq_id = sample.sequence_id
        ref = sample.activations

        kin = run_kinesis_from_artifact(k_manifest[seq_id]["path"], seq_id, sample.text)
        if collect_timings:
            kin_times.append(float(k_manifest[seq_id].get("timing_s", float("nan"))))

        kin_pred = kin.activations[:, mapping.kinesis_indices]
        kin_ref = ref[:, mapping.rajagopal_indices]
        kin_pred, kin_ref = align_lengths(kin_pred, kin_ref)
        per_sample.append(
            compute_l1_metrics(kin_pred, kin_ref, mapping.shared_names, **paired_l1_kw)
        )

    return per_sample, kin_times


def aggregate_paired_layer1(
    mm_samples: list[dict[str, Any]],
    kin_samples: list[dict[str, Any]],
    mapping: MuscleMapping,
) -> dict[str, Any]:
    """Build a ``layer1_paired`` results block."""
    if not mm_samples or len(mm_samples) != len(kin_samples):
        return {
            "error": "paired sample count mismatch",
            "n_musclemap": len(mm_samples),
            "n_kinesis": len(kin_samples),
        }
    return {
        "n_sequences": len(mm_samples),
        "n_muscles": len(mapping.shared_names),
        "muscle_names": list(mapping.shared_names),
        "description": (
            "Paired comparison on sequences with Kinesis precompute artifacts; "
            "both methods scored on the same mapped Rajagopal leg muscles."
        ),
        "musclemap": aggregate_l1_metrics(mm_samples),
        "kinesis": aggregate_l1_metrics(kin_samples),
    }


def find_first_index_for_sequence(dataset: Any, sequence_id: str) -> int | None:
    """First test-split window index for a sequence id, or None."""
    for idx in range(len(dataset)):
        if _sequence_id_for_index(dataset, idx) == sequence_id:
            return idx
    return None


def load_smplx_window(dataset: Any, index: int) -> np.ndarray:
    """SMPL-X motion slice for one MuscleActivationDataset window."""
    seq_dir, start, true_T, _text = dataset._items[index]
    motion = np.load(seq_dir / "smplx_322.npy").astype(np.float32)
    return motion[int(start) : int(start) + int(true_T)]


def expand_activations_to_rajagopal(
    mapped: np.ndarray,
    raj_indices: list[int],
    n_muscles: int = 80,
) -> np.ndarray:
    """Scatter mapped leg activations into a full Rajagopal vector."""
    out = np.zeros((mapped.shape[0], n_muscles), dtype=np.float32)
    for j, r_idx in enumerate(raj_indices):
        out[:, int(r_idx)] = mapped[:, j]
    return out


def build_triple_vis_example(
    sequence_id: str,
    *,
    cfg: dict[str, Any],
    dataset: Any,
    model: Any,
    k_manifest: dict[str, dict[str, Any]],
    mapping: MuscleMapping,
    rajagopal_names: list[str],
    device: str,
) -> dict[str, Any]:
    """Build GT / MuscleMAP / Kinesis arrays for one sequence (for notebook animation)."""
    if sequence_id not in k_manifest:
        raise ValueError(f"No Kinesis artifact for {sequence_id!r}")

    idx = find_first_index_for_sequence(dataset, sequence_id)
    if idx is not None:
        sample = get_benchmark_sample(dataset, idx)
        ref = sample.activations
        text = sample.text
        smplx_win = load_smplx_window(dataset, idx)
    else:
        from src.body_visualization import find_sequence_dir
        from src.loaders import _resolve_path
        from src.test_split import clean_label, discover_test_split_sequence_entries

        ent: dict[str, Any] | None = None
        entries = {
            e["seq_id"]: e
            for e in discover_test_split_sequence_entries(cfg, bench_root=_bench_root())
        }
        if sequence_id in entries:
            ent = entries[sequence_id]
        else:
            ds_root = _resolve_path(cfg["test_set"]["dataset_root"])
            seq_dir = find_sequence_dir(ds_root, sequence_id)
            label_path = seq_dir / "semantic_label.txt"
            text_disk = (
                label_path.read_text(encoding="utf-8").strip()
                if label_path.is_file()
                else clean_label(sequence_id)
            )
            ent = {
                "seq_id": sequence_id,
                "text": text_disk,
                "smplx_npy": str((seq_dir / "smplx_322.npy").resolve()),
                "activations_npy": str((seq_dir / "activations.npy").resolve()),
            }

        text = ent["text"]
        ref = np.load(ent["activations_npy"]).astype(np.float32)
        motion = np.load(ent["smplx_npy"]).astype(np.float32)
        n = min(ref.shape[0], motion.shape[0])
        ref = ref[:n]
        smplx_win = motion[:n]

    mm = run_musclemap(model, text, sequence_id, device=device, ref_T=ref.shape[0])
    kin = run_kinesis_from_artifact(k_manifest[sequence_id]["path"], sequence_id, text)

    mm_pred, ref_a = align_lengths(mm.activations, ref)
    kin_pred = kin.activations[:, mapping.kinesis_indices]
    kin_ref = ref[:, mapping.rajagopal_indices]
    kin_pred, _kin_ref = align_lengths(kin_pred, kin_ref)

    T = min(smplx_win.shape[0], mm_pred.shape[0], kin_pred.shape[0], ref_a.shape[0])
    return {
        "sequence_id": sequence_id,
        "prompt": text,
        "smplx": smplx_win[:T],
        "gt": ref_a[:T],
        "mm": mm_pred[:T],
        "kin_full": expand_activations_to_rajagopal(
            kin_pred[:T], mapping.rajagopal_indices, len(rajagopal_names)
        ),
    }


def ensure_vis_examples(
    examples: list[dict[str, Any]],
    sequence_id: str,
    *,
    max_vis: int,
    cfg: dict[str, Any],
    dataset: Any,
    model: Any,
    k_manifest: dict[str, dict[str, Any]],
    mapping: MuscleMapping,
    rajagopal_names: list[str],
    device: str,
) -> list[dict[str, Any]]:
    """Append a visualization example for ``sequence_id`` if missing."""
    if any(ex.get("sequence_id") == sequence_id for ex in examples):
        return examples
    if len(examples) >= max_vis:
        return examples
    examples.append(
        build_triple_vis_example(
            sequence_id,
            cfg=cfg,
            dataset=dataset,
            model=model,
            k_manifest=k_manifest,
            mapping=mapping,
            rajagopal_names=rajagopal_names,
            device=device,
        )
    )
    return examples
