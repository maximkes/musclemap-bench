from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from src.align import dtw_align


def _pearson_per_muscle(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    x = pred - pred.mean(axis=0, keepdims=True)
    y = ref - ref.mean(axis=0, keepdims=True)
    num = (x * y).sum(axis=0)
    den = np.sqrt((x**2).sum(axis=0) * (y**2).sum(axis=0)) + 1e-8
    return (num / den).astype(np.float32)


def _r2_per_muscle(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    ss_res = ((pred - ref) ** 2).sum(axis=0)
    ss_tot = ((ref - ref.mean(axis=0)) ** 2).sum(axis=0) + 1e-8
    return (1.0 - ss_res / ss_tot).astype(np.float32)


def _onset_frames(acts: np.ndarray, threshold: float, min_frames: int) -> np.ndarray:
    T, N = acts.shape
    out = np.full(N, -1, dtype=np.int32)
    for n in range(N):
        for t in range(T - min_frames + 1):
            if np.all(acts[t:t + min_frames, n] >= threshold):
                out[n] = t
                break
    return out


def compute_l1_metrics(
    pred: np.ndarray,
    ref: np.ndarray,
    muscle_names: Sequence[str],
    *,
    dtw_enabled: bool = True,
    dtw_radius: int = 10,
    activation_threshold: float = 0.2,
    min_active_frames: int = 3,
    compute_energy: bool = True,
    compute_coactivation: bool = True,
) -> dict[str, Any]:
    pred = np.asarray(pred, dtype=np.float32)
    ref = np.asarray(ref, dtype=np.float32)
    if pred.shape != ref.shape:
        raise ValueError(f"pred/ref shape mismatch: {pred.shape} vs {ref.shape}")
    T, N = pred.shape
    if len(muscle_names) != N:
        raise ValueError(f"muscle_names length {len(muscle_names)} != N={N}")

    diff = pred - ref
    out: dict[str, Any] = {
        "n_frames": T,
        "n_muscles": N,
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
    }

    if dtw_enabled:
        pred_w, ref_w = dtw_align(pred, ref, radius=dtw_radius)
        diff_w = pred_w - ref_w
        out["dtw_mae"] = float(np.mean(np.abs(diff_w)))
        out["dtw_rmse"] = float(np.sqrt(np.mean(diff_w ** 2)))
        out["dtw_path_length"] = int(len(pred_w))
    else:
        out["dtw_mae"] = None
        out["dtw_rmse"] = None

    pr = _pearson_per_muscle(pred, ref)
    r2 = _r2_per_muscle(pred, ref)
    out["pearson_r_mean"] = float(np.mean(pr))
    out["pearson_r_per_muscle"] = pr.tolist()
    out["r2_mean"] = float(np.mean(r2))
    out["r2_per_muscle"] = r2.tolist()
    out["r2_per_muscle_named"] = {name: float(val) for name, val in zip(muscle_names, r2)}

    pred_on = _onset_frames(pred, activation_threshold, min_active_frames)
    ref_on = _onset_frames(ref, activation_threshold, min_active_frames)
    valid = (pred_on >= 0) & (ref_on >= 0)
    if np.any(valid):
        onset_err = np.abs(pred_on[valid] - ref_on[valid]).astype(np.float32)
        out["onset_timing_error_mean"] = float(np.mean(onset_err))
        out["onset_timing_error_std"] = float(np.std(onset_err))
        out["onset_timing_n_compared_muscles"] = int(np.sum(valid))
        out["onset_timing_errors"] = onset_err.tolist()
    else:
        out["onset_timing_error_mean"] = None
        out["onset_timing_error_std"] = None
        out["onset_timing_n_compared_muscles"] = 0
        out["onset_timing_errors"] = []

    if compute_energy:
        out["energy_pred"] = float(np.mean(pred ** 2))
        out["energy_ref"] = float(np.mean(ref ** 2))
        out["energy_ratio"] = float(out["energy_pred"] / out["energy_ref"]) if out["energy_ref"] > 1e-9 else None

    if compute_coactivation:
        def _co(a: np.ndarray) -> np.ndarray:
            ac = a - a.mean(axis=0, keepdims=True)
            norms = np.sqrt((ac ** 2).sum(axis=0)) + 1e-8
            return (ac.T @ ac) / np.outer(norms, norms)
        out["coactivation_frobenius"] = float(np.linalg.norm(_co(pred) - _co(ref), ord="fro"))

    out["smoothness_pred"] = float(np.mean(np.abs(np.diff(pred, axis=0)))) if T > 1 else 0.0
    out["smoothness_ref"] = float(np.mean(np.abs(np.diff(ref, axis=0)))) if T > 1 else 0.0
    return out


def aggregate_l1_metrics(per_sample: list[dict[str, Any]]) -> dict[str, Any]:
    if not per_sample:
        return {}
    scalar_keys = [
        "mae", "rmse", "dtw_mae", "dtw_rmse", "pearson_r_mean", "r2_mean",
        "onset_timing_error_mean", "energy_pred", "energy_ref", "energy_ratio",
        "coactivation_frobenius", "smoothness_pred", "smoothness_ref",
    ]
    out: dict[str, Any] = {"n_samples": len(per_sample)}
    for key in scalar_keys:
        vals = [x[key] for x in per_sample if x.get(key) is not None]
        out[key] = float(np.mean(vals)) if vals else None

    first_named = per_sample[0].get("r2_per_muscle_named", {})
    if first_named:
        keys = list(first_named.keys())
        out["r2_per_muscle_named"] = {
            k: float(np.mean([sample["r2_per_muscle_named"][k] for sample in per_sample]))
            for k in keys
        }

    onset_errors: list[float] = []
    for sample in per_sample:
        onset_errors.extend(sample.get("onset_timing_errors", []))
    out["onset_timing_errors_all"] = onset_errors
    return out
