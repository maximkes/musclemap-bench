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


def compute_resource_summary(musclemap_timings: list[float], kinesis_timings: list[float], motiongpt_timings: list[float], cfg: dict[str, Any]) -> dict[str, Any]:
    r = cfg["resources"]
    return {
        "musclemap": {
            "inference": _stats(musclemap_timings),
            "training_gpus": r["musclemap_training_gpus"],
            "training_gpu_type": r["musclemap_training_gpu_type"],
            "training_hours": r["musclemap_training_hours"],
            "training_gpu_hours": (r["musclemap_training_gpus"] * r["musclemap_training_hours"]) if r["musclemap_training_hours"] else None,
        },
        "kinesis": {
            "inference": _stats(kinesis_timings),
            "training_gpus": 0,
            "training_hours": 0,
            "training_gpu_hours": 0,
            "note": "Physics solver baseline; no training cost.",
        },
        "motiongpt": {
            "inference": _stats(motiongpt_timings),
            "training_gpus": r["motiongpt_training_gpus"],
            "training_gpu_type": r["motiongpt_training_gpu_type"],
            "training_hours": r["motiongpt_training_hours"],
            "training_gpu_hours": (r["motiongpt_training_gpus"] * r["motiongpt_training_hours"]) if r["motiongpt_training_hours"] else None,
        },
    }
