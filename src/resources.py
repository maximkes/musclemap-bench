from __future__ import annotations

from typing import Any

import numpy as np


def _stats(ts: list[float]) -> dict[str, float | None]:
    arr = [float(x) for x in ts if x == x]
    if not arr:
        return {"mean_s": None, "std_s": None, "median_s": None}
    return {
        "mean_s": float(np.mean(arr)),
        "std_s": float(np.std(arr)),
        "median_s": float(np.median(arr)),
    }


def _training_gpu_hours(gpus: int, hours: float | int | None) -> float | None:
    if hours is None:
        return None
    return float(gpus) * float(hours)


def compute_resource_summary(musclemap_timings: list[float], kinesis_timings: list[float], motiongpt_timings: list[float], cfg: dict[str, Any]) -> dict[str, Any]:
    r = cfg["resources"]
    kin_gpus = int(r.get("kinesis_training_gpus", 1))
    kin_hours = r.get("kinesis_training_hours")
    kin_gpu_hours = _training_gpu_hours(kin_gpus, kin_hours)
    return {
        "musclemap": {
            "inference": _stats(musclemap_timings),
            "inference_note": "Wall-clock on test-paired windows only (see profile_paired_inference.py).",
            "training_gpus": r["musclemap_training_gpus"],
            "training_gpu_type": r["musclemap_training_gpu_type"],
            "training_hours": r["musclemap_training_hours"],
            "training_gpu_hours": _training_gpu_hours(r["musclemap_training_gpus"], r["musclemap_training_hours"]),
        },
        "kinesis": {
            "inference": _stats(kinesis_timings),
            "inference_note": "Precompute simulation wall-clock on test-paired windows only.",
            "training_gpus": kin_gpus,
            "training_gpu_type": r.get("kinesis_training_gpu_type"),
            "training_hours": kin_hours,
            "training_gpu_hours": kin_gpu_hours,
            "note": (
                "Kinesis RL training (~10 days on 1× A100 per arXiv:2503.14637); "
                "inference here is physics simulation from precomputed artifacts."
            ),
        },
        "motiongpt": {
            "inference": _stats(motiongpt_timings),
            "training_gpus": r["motiongpt_training_gpus"],
            "training_gpu_type": r["motiongpt_training_gpu_type"],
            "training_hours": r["motiongpt_training_hours"],
            "training_gpu_hours": _training_gpu_hours(r["motiongpt_training_gpus"], r["motiongpt_training_hours"]),
        },
    }
