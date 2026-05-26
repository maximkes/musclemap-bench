from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.interpolate import interp1d


@dataclass
class MuscleMapping:
    pairs: list[tuple[int, int]] = field(default_factory=list)
    rajagopal_names: list[str] = field(default_factory=list)
    kinesis_names: list[str] = field(default_factory=list)

    @property
    def rajagopal_indices(self) -> list[int]:
        return [p[0] for p in self.pairs]

    @property
    def kinesis_indices(self) -> list[int]:
        return [p[1] for p in self.pairs]

    @property
    def shared_names(self) -> list[str]:
        return [self.rajagopal_names[p[0]] for p in self.pairs]

    def __len__(self) -> int:
        return len(self.pairs)


def build_muscle_mapping(mapping_json: str | Path, rajagopal_names: Sequence[str], kinesis_names: Sequence[str]) -> MuscleMapping:
    raw = json.loads(Path(mapping_json).read_text(encoding="utf-8"))
    entries = raw["mapping"]
    raj_index = {n: i for i, n in enumerate(rajagopal_names)}
    kin_index = {n: i for i, n in enumerate(kinesis_names)}

    pairs: list[tuple[int, int]] = []
    for entry in entries:
        raj_name = entry["rajagopal"]
        kin_name = entry.get("kinesis")
        if kin_name is None:
            continue
        if raj_name not in raj_index or kin_name not in kin_index:
            continue
        pairs.append((raj_index[raj_name], kin_index[kin_name]))

    return MuscleMapping(pairs=pairs, rajagopal_names=list(rajagopal_names), kinesis_names=list(kinesis_names))


def build_mapping_by_shared_names(
    rajagopal_names: Sequence[str],
    kinesis_names: Sequence[str],
) -> MuscleMapping:
    """Pair muscles with identical names in the dataset and Kinesis artifact columns."""
    raj_index = {n: i for i, n in enumerate(rajagopal_names)}
    pairs: list[tuple[int, int]] = []
    for ki, kin_name in enumerate(kinesis_names):
        ri = raj_index.get(kin_name)
        if ri is not None:
            pairs.append((ri, ki))
    return MuscleMapping(
        pairs=pairs,
        rajagopal_names=list(rajagopal_names),
        kinesis_names=list(kinesis_names),
    )


def build_mapping_kinesis_artifacts(
    rajagopal_names: Sequence[str],
    kinesis_artifact_names: Sequence[str],
    artifact_to_dataset: dict[str, str],
) -> MuscleMapping:
    """Map Kinesis precompute column names to dataset Rajagopal indices."""
    raj_index = {n: i for i, n in enumerate(rajagopal_names)}
    pairs: list[tuple[int, int]] = []
    for ki, art_name in enumerate(kinesis_artifact_names):
        dataset_name = artifact_to_dataset.get(art_name, art_name)
        ri = raj_index.get(dataset_name)
        if ri is None:
            ri = raj_index.get(art_name)
        if ri is not None:
            pairs.append((ri, ki))
    return MuscleMapping(
        pairs=pairs,
        rajagopal_names=list(rajagopal_names),
        kinesis_names=list(kinesis_artifact_names),
    )


def resample_to_length(arr: np.ndarray, target_T: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    T, _ = arr.shape
    if T == target_T:
        return arr.copy()
    x_old = np.linspace(0.0, 1.0, T)
    x_new = np.linspace(0.0, 1.0, target_T)
    interp = interp1d(x_old, arr, axis=0, kind="linear", fill_value="extrapolate")
    return interp(x_new).astype(np.float32)


def align_lengths(a: np.ndarray, b: np.ndarray, resample: bool = True) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"feature dimension mismatch: {a.shape} vs {b.shape}")
    if a.shape[0] == b.shape[0]:
        return a.copy(), b.copy()
    if resample:
        target = min(a.shape[0], b.shape[0])
        return resample_to_length(a, target), resample_to_length(b, target)
    target = min(a.shape[0], b.shape[0])
    return a[:target].copy(), b[:target].copy()


def dtw_align(pred: np.ndarray, ref: np.ndarray, radius: int = 10) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred, dtype=np.float32)
    ref = np.asarray(ref, dtype=np.float32)
    try:
        from fastdtw import fastdtw
    except ModuleNotFoundError:
        # Safe fallback for test and minimal environments: return identity alignment.
        # Full benchmark environments should install fastdtw.
        return pred.copy(), ref.copy()
    _, path = fastdtw(pred, ref, radius=radius)
    pred_idx = [i for i, _ in path]
    ref_idx = [j for _, j in path]
    return pred[pred_idx].astype(np.float32), ref[ref_idx].astype(np.float32)
