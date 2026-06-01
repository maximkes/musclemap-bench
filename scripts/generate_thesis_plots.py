#!/usr/bin/env python3
"""Regenerate thesis figure PNGs (test-paired metrics only)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.curve_data import interpolate_curve
from src.loaders import _resolve_path, load_config


def _norm_gpu_hours(gpus, hours, gpu_type, *, ref_gpu, perf):
    scale = float(perf.get(gpu_type, 1.0)) / float(perf.get(ref_gpu, 1.0))
    return float(gpus) * float(hours) * scale


def _load_kinesis_baseline(cfg, results: dict) -> tuple[float, float]:
    paired = results.get("layer1_paired", {})
    kin = paired.get("kinesis", {})
    if kin.get("mae") is not None:
        return float(kin["mae"]), float(kin.get("rmse", np.nan))
    baseline = cfg["baselines"]["kinesis_test_paired"]
    return float(baseline["mae"]), float(baseline["rmse"])


def main() -> None:
    cfg = load_config(ROOT / "config.yaml")
    plots = _resolve_path(cfg["paths"]["results_dir"]) / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    results_path = _resolve_path(cfg["paths"]["results_dir"]) / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else {}

    paired_csv = _resolve_path(cfg["paths"]["curves_dir"]) / "test_paired_metrics.csv"
    if not paired_csv.exists():
        print(f"Missing {paired_csv} — run scripts/build_test_paired_curves.py first")
        sys.exit(1)

    df = pd.read_csv(paired_csv).sort_values("epoch").reset_index(drop=True)
    kin_mae, kin_rmse = _load_kinesis_baseline(cfg, results)

    step_grid = np.linspace(df["global_step"].min(), df["global_step"].max(), 200)
    n_ckpt = len(df)
    window = min(7, n_ckpt if n_ckpt % 2 else max(n_ckpt - 1, 3))
    if window % 2 == 0:
        window -= 1
    window = max(window, 3)
    poly = min(2, window - 1)
    mae_smooth = savgol_filter(
        interpolate_curve(df, step_grid, value_col="mae"), window_length=window, polyorder=poly
    )
    rmse_smooth = savgol_filter(
        interpolate_curve(df, step_grid, value_col="rmse"), window_length=window, polyorder=poly
    )
    best = df.loc[df["mae"].idxmin()]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for ax, metric, smooth, kin_v, ylab in zip(
        axes,
        ("mae", "rmse"),
        (mae_smooth, rmse_smooth),
        (kin_mae, kin_rmse),
        ("Paired test MAE (42 muscles)", "Paired test RMSE (42 muscles)"),
    ):
        ax.scatter(df["global_step"], df[metric], color="#4C72B0", s=36, alpha=0.85, label="MuscleMAP checkpoints")
        ax.plot(step_grid, smooth, color="#4C72B0", linewidth=1.8, label="MuscleMAP (smoothed)")
        ax.axhline(kin_v, color="#C44E52", linestyle="--", label="Kinesis (paired baseline)")
        ax.scatter(
            [best["global_step"]],
            [best[metric]],
            s=120,
            color="#55A868",
            label=f"Best ep {int(best['epoch'])}",
        )
        ax.set_ylabel(ylab)
        ax.set_xlabel("Optimizer step")
        ax.legend(fontsize=7)
    n_paired = int(df["n_paired_windows"].iloc[0])
    fig.suptitle(f"Test-paired metrics ({n_paired} windows, 42 leg muscles)")
    fig.tight_layout()
    p_a = plots / "training_mae_rmse_vs_step.png"
    fig.savefig(p_a, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    paired = results.get("layer1_paired", {})
    mm_mae = paired.get("musclemap", {}).get("mae", best["mae"])
    ax.bar(["Kinesis", "MuscleMAP"], [kin_mae, float(mm_mae)], color=["#C44E52", "#4C72B0"])
    ax.set_ylabel("Paired test MAE (42 muscles)")
    ax.set_title(f"Best checkpoint ep {int(best['epoch'])} vs Kinesis")
    ax.axhline(kin_mae * 1.05, color="gray", linestyle=":", label="Kinesis × 1.05")
    ax.legend(fontsize=8)
    for i, v in enumerate([kin_mae, float(mm_mae)]):
        ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    p_bar = plots / "test_paired_mae_vs_kinesis.png"
    fig.savefig(p_bar, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for ax, col, ylab in ((axes[0], "mae", "MAE"), (axes[1], "rmse", "RMSE")):
        ax.plot(df["epoch"], df[col], marker="o", color="#4C72B0", label=f"MuscleMAP {ylab}")
        kin_col = kin_mae if col == "mae" else kin_rmse
        ax.axhline(float(kin_col), color="#C44E52", linestyle="--", label="Kinesis paired")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylab)
        ax.legend(fontsize=8)
    fig.tight_layout()
    p_ckpt = plots / "test_paired_metrics_by_checkpoint.png"
    fig.savefig(p_ckpt, dpi=150, bbox_inches="tight")
    plt.close(fig)

    r_cfg = cfg["resources"]
    ref_gpu = r_cfg.get("compute_reference_gpu", "A100")
    perf = r_cfg.get("gpu_relative_performance", {"V100-16GB": 0.52, "A100": 1.0})
    mm_h = float(r_cfg.get("musclemap_training_hours") or 14)
    mm_norm = _norm_gpu_hours(
        r_cfg["musclemap_training_gpus"],
        mm_h,
        r_cfg.get("musclemap_training_gpu_type", "V100-16GB"),
        ref_gpu=ref_gpu,
        perf=perf,
    )
    kin_norm = _norm_gpu_hours(
        r_cfg.get("kinesis_training_gpus", 1),
        r_cfg.get("kinesis_training_hours"),
        r_cfg.get("kinesis_training_gpu_type", "A100"),
        ref_gpu=ref_gpu,
        perf=perf,
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    step_max = float(step_grid.max())
    mm_compute_h = mm_norm * (np.asarray(step_grid, dtype=np.float64) / step_max)
    ax.plot(mm_compute_h, mae_smooth, color="#4C72B0", linewidth=1.8, label="MuscleMAP (paired test MAE)")
    ax.scatter([kin_norm], [kin_mae], s=120, color="#C44E52", label="Kinesis (paired baseline)")
    best_h = mm_norm * (float(best["global_step"]) / step_max)
    ax.scatter([best_h], [float(best["mae"])], s=120, color="#55A868", label=f"Best ep {int(best['epoch'])}")
    ax.set_xscale("log")
    ax.set_xlabel(f"Training compute ({ref_gpu}-eq GPU-hours, log)")
    ax.set_ylabel("Paired test MAE (42 muscles)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    p_c = plots / "training_compute_vs_mae.png"
    fig.savefig(p_c, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {p_a}")
    print(f"Wrote {p_bar}")
    print(f"Wrote {p_ckpt}")
    print(f"Wrote {p_c}")
    print(f"Best paired ep {int(best['epoch'])} MAE={best['mae']:.4f} vs Kinesis {kin_mae:.4f}")


if __name__ == "__main__":
    main()
