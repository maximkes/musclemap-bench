#!/usr/bin/env python3
"""Rebuild thesis_figures.ipynb (source-only, test-paired metrics)."""
from __future__ import annotations

import json
from pathlib import Path


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [text],
    }


NB = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "musclemap-model", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "cells": [
        md(
            "# Thesis figures (musclemap-bench)\n\n"
            "Run with working directory = `musclemap-bench`.\n\n"
            "Prerequisites:\n"
            "- `python precompute/run_kinesis.py --config config.yaml --test-split-only`\n"
            "- `python scripts/build_test_paired_curves.py --config config.yaml`\n"
            "- `python scripts/run_benchmark.py --config config.yaml --export-timings`\n"
            "- `python scripts/profile_paired_inference.py`\n\n"
            "All accuracy plots use **test-paired** metrics only (windows with Kinesis artifacts, 42 leg muscles)."
        ),
        code(
            "from __future__ import annotations\n\n"
            "import json\n"
            "from pathlib import Path\n\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n"
            "import pandas as pd\n\n"
            "from IPython.display import display, Markdown\n\n"
            "from src.curve_data import interpolate_curve\n"
            "from src.loaders import load_config, _resolve_path\n\n"
            "REPO_ROOT = Path.cwd()\n"
            "if not (REPO_ROOT / \"config.yaml\").exists() and (REPO_ROOT.parent / \"config.yaml\").exists():\n"
            "    REPO_ROOT = REPO_ROOT.parent\n\n"
            "cfg = load_config(REPO_ROOT / \"config.yaml\")\n"
            "RESULTS_DIR = _resolve_path(cfg[\"paths\"][\"results_dir\"])\n"
            "PLOTS_DIR = RESULTS_DIR / \"plots\"\n"
            "PLOTS_DIR.mkdir(parents=True, exist_ok=True)\n\n"
            "results_path = RESULTS_DIR / \"results.json\"\n"
            "results = json.loads(results_path.read_text(encoding=\"utf-8\")) if results_path.exists() else {}\n\n"
            "paired_csv = _resolve_path(cfg[\"paths\"][\"curves_dir\"]) / \"test_paired_metrics.csv\"\n"
            "if not paired_csv.exists():\n"
            "    raise FileNotFoundError(\"Run scripts/build_test_paired_curves.py first\")\n"
            "df = pd.read_csv(paired_csv).sort_values(\"epoch\").reset_index(drop=True)\n\n"
            "paired = results.get(\"layer1_paired\", {})\n"
            "kin_mae = paired.get(\"kinesis\", {}).get(\"mae\") or cfg[\"baselines\"][\"kinesis_test_paired\"][\"mae\"]\n"
            "kin_rmse = paired.get(\"kinesis\", {}).get(\"rmse\") or cfg[\"baselines\"][\"kinesis_test_paired\"][\"rmse\"]\n"
            "n_paired = int(df[\"n_paired_windows\"].iloc[0])\n\n"
            "plt.style.use(\"seaborn-v0_8-whitegrid\")\n"
            "print(f\"Repo: {REPO_ROOT}; paired windows: {n_paired}\")"
        ),
        md("## Figure A — Test-paired MAE/RMSE vs optimizer step"),
        code(
            "from scipy.signal import savgol_filter\n\n"
            "step_min, step_max = int(df[\"global_step\"].min()), int(df[\"global_step\"].max())\n"
            "step_grid = np.linspace(step_min, step_max, 200)\n"
            "n_ckpt = len(df)\n"
            "window = min(7, n_ckpt if n_ckpt % 2 else max(n_ckpt - 1, 3))\n"
            "if window % 2 == 0:\n"
            "    window -= 1\n"
            "window = max(window, 3)\n"
            "poly = min(2, window - 1)\n"
            "mae_smooth = savgol_filter(interpolate_curve(df, step_grid, value_col=\"mae\"), window_length=window, polyorder=poly)\n"
            "rmse_smooth = savgol_filter(interpolate_curve(df, step_grid, value_col=\"rmse\"), window_length=window, polyorder=poly)\n"
            "best_ep = df.loc[df[\"mae\"].idxmin()]\n\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)\n"
            "for ax, metric, smooth, kin_val, ylab in zip(\n"
            "    axes,\n"
            "    (\"mae\", \"rmse\"),\n"
            "    (mae_smooth, rmse_smooth),\n"
            "    (float(kin_mae), float(kin_rmse)),\n"
            "    (\"Paired test MAE (42 muscles)\", \"Paired test RMSE (42 muscles)\"),\n"
            "):\n"
            "    ax.scatter(df[\"global_step\"], df[metric], color=\"#4C72B0\", s=36, alpha=0.85, label=\"Checkpoints\")\n"
            "    ax.plot(step_grid, smooth, color=\"#4C72B0\", linewidth=1.8, label=\"Smoothed\")\n"
            "    ax.axhline(kin_val, color=\"#C44E52\", linestyle=\"--\", linewidth=1.2, label=\"Kinesis baseline\")\n"
            "    ax.scatter([best_ep[\"global_step\"]], [best_ep[metric]], s=120, color=\"#55A868\", label=f\"Best ep {int(best_ep['epoch'])}\")\n"
            "    ax.set_ylabel(ylab)\n"
            "    ax.set_xlabel(\"Optimizer step (global)\")\n"
            "    ax.legend(fontsize=7, loc=\"upper right\")\n\n"
            "fig.suptitle(f\"Test-paired metrics ({n_paired} windows)\", y=1.03, fontsize=11)\n"
            "fig.tight_layout()\n"
            "out_a = PLOTS_DIR / \"training_mae_rmse_vs_step.png\"\n"
            "fig.savefig(out_a, dpi=150, bbox_inches=\"tight\")\n"
            "plt.show()\n"
            "print(f\"Wrote {out_a}; best ep {int(best_ep['epoch'])} MAE={best_ep['mae']:.4f}\")"
        ),
        code(
            "mm_mae = paired.get(\"musclemap\", {}).get(\"mae\", best_ep[\"mae\"])\n"
            "fig, ax = plt.subplots(figsize=(5, 4))\n"
            "ax.bar([\"Kinesis\", \"MuscleMAP\"], [float(kin_mae), float(mm_mae)], color=[\"#C44E52\", \"#4C72B0\"])\n"
            "ax.set_ylabel(\"Paired test MAE (42 muscles)\")\n"
            "ax.set_title(f\"Best checkpoint ep {int(best_ep['epoch'])} vs Kinesis\")\n"
            "ax.axhline(float(kin_mae) * 1.05, color=\"gray\", linestyle=\":\", label=\"Kinesis × 1.05\")\n"
            "ax.legend(fontsize=8)\n"
            "for i, v in enumerate([kin_mae, mm_mae]):\n"
            "    ax.text(i, v, f\"{float(v):.4f}\", ha=\"center\", va=\"bottom\", fontsize=9)\n"
            "fig.tight_layout()\n"
            "out_test = PLOTS_DIR / \"test_paired_mae_vs_kinesis.png\"\n"
            "fig.savefig(out_test, dpi=150, bbox_inches=\"tight\")\n"
            "plt.show()\n"
            "display(Markdown(f\"**Paired:** Kinesis {kin_mae:.4f}, MuscleMAP {float(mm_mae):.4f} ({(float(mm_mae)-kin_mae)/kin_mae*100:+.1f}%).\"))"
        ),
        md("### Checkpoint table (test-paired)"),
        code(
            "summary = df[[\"stem\", \"epoch\", \"global_step\", \"mae\", \"rmse\", \"n_paired_windows\"]].copy()\n"
            "summary[\"mae_vs_kinesis_pct\"] = (summary[\"mae\"] - kin_mae) / kin_mae * 100\n"
            "display(summary.style.format({\"mae\": \"{:.4f}\", \"rmse\": \"{:.4f}\", \"mae_vs_kinesis_pct\": \"{:+.1f}%\"}))\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)\n"
            "for ax, col, ylab in ((axes[0], \"mae\", \"MAE\"), (axes[1], \"rmse\", \"RMSE\")):\n"
            "    ax.plot(summary[\"epoch\"], summary[col], marker=\"o\", color=\"#4C72B0\", label=f\"MuscleMAP {ylab}\")\n"
            "    kin_col = kin_mae if col == \"mae\" else kin_rmse\n"
            "    ax.axhline(float(kin_col), color=\"#C44E52\", linestyle=\"--\", label=\"Kinesis\")\n"
            "    ax.set_xlabel(\"Epoch\")\n"
            "    ax.set_ylabel(ylab)\n"
            "    ax.legend(fontsize=8)\n"
            "fig.tight_layout()\n"
            "out_ckpt = PLOTS_DIR / \"test_paired_metrics_by_checkpoint.png\"\n"
            "fig.savefig(out_ckpt, dpi=150, bbox_inches=\"tight\")\n"
            "plt.show()\n"
            "print(f\"Wrote {out_ckpt}\")"
        ),
        md("## Figure B — Inference time (paired windows only)"),
        code(
            "profile_path = RESULTS_DIR / \"inference_profile.json\"\n"
            "profile = json.loads(profile_path.read_text(encoding=\"utf-8\")) if profile_path.exists() else {}\n"
            "paired_inf = profile.get(\"paired_inference\", {})\n"
            "mm_t = paired_inf.get(\"musclemap\", results.get(\"resources\", {}).get(\"musclemap\", {}).get(\"inference\", {}))\n"
            "kin_t = paired_inf.get(\"kinesis\", results.get(\"resources\", {}).get(\"kinesis\", {}).get(\"inference\", {}))\n"
            "labels, means = [\"MuscleMAP\", \"Kinesis\"], [mm_t.get(\"mean_s\"), kin_t.get(\"mean_s\")]\n"
            "if all(m is not None for m in means):\n"
            "    fig, ax = plt.subplots(figsize=(5, 4))\n"
            "    ax.bar(labels, [float(m) for m in means], color=[\"#4C72B0\", \"#C44E52\"])\n"
            "    ax.set_ylabel(\"Mean inference time (s / paired window)\")\n"
            "    ax.set_title(f\"Inference latency (n={paired_inf.get('n_paired_windows', n_paired)} paired windows)\")\n"
            "    fig.tight_layout()\n"
            "    out_b = PLOTS_DIR / \"inference_time_mean.png\"\n"
            "    fig.savefig(out_b, dpi=150, bbox_inches=\"tight\")\n"
            "    plt.show()\n"
            "    print(f\"Wrote {out_b}\")\n"
            "else:\n"
            "    print(\"Run scripts/profile_paired_inference.py and scripts/run_benchmark.py --export-timings\")"
        ),
        md("## Figure C — Training compute vs paired test MAE"),
        code(
            "def _norm_gpu_hours(gpus, hours, gpu_type, *, ref_gpu, perf):\n"
            "    if hours is None:\n"
            "        return None\n"
            "    scale = float(perf.get(gpu_type, 1.0)) / float(perf.get(ref_gpu, 1.0))\n"
            "    return float(gpus) * float(hours) * scale\n\n"
            "r_cfg = cfg[\"resources\"]\n"
            "ref_gpu = r_cfg.get(\"compute_reference_gpu\", \"A100\")\n"
            "perf = r_cfg.get(\"gpu_relative_performance\", {\"V100-16GB\": 0.52, \"A100\": 1.0, \"A100-80GB\": 1.0})\n"
            "mm_hours = r_cfg.get(\"musclemap_training_hours\") or 14\n"
            "mm_norm_h = _norm_gpu_hours(r_cfg[\"musclemap_training_gpus\"], mm_hours, r_cfg.get(\"musclemap_training_gpu_type\", \"V100-16GB\"), ref_gpu=ref_gpu, perf=perf)\n"
            "kin_norm_h = _norm_gpu_hours(r_cfg.get(\"kinesis_training_gpus\", 1), r_cfg.get(\"kinesis_training_hours\"), r_cfg.get(\"kinesis_training_gpu_type\", \"A100\"), ref_gpu=ref_gpu, perf=perf)\n"
            "step_max = float(np.max(step_grid))\n"
            "mm_compute_h = mm_norm_h * (np.asarray(step_grid, dtype=np.float64) / step_max)\n"
            "best_compute_h = float(mm_norm_h * (float(best_ep[\"global_step\"]) / step_max))\n"
            "fig, ax = plt.subplots(figsize=(7, 4.5))\n"
            "ax.plot(mm_compute_h, mae_smooth, color=\"#4C72B0\", linewidth=1.8, label=\"MuscleMAP paired MAE\")\n"
            "ax.scatter([kin_norm_h], [float(kin_mae)], s=120, color=\"#C44E52\", label=\"Kinesis baseline\")\n"
            "ax.scatter([best_compute_h], [float(best_ep[\"mae\"])], s=120, color=\"#55A868\", label=f\"Best ep {int(best_ep['epoch'])}\")\n"
            "ax.set_xscale(\"log\")\n"
            "ax.set_xlabel(f\"Training compute ({ref_gpu}-equivalent GPU-hours, log)\")\n"
            "ax.set_ylabel(\"Paired test MAE (42 muscles)\")\n"
            "ax.legend(fontsize=8, loc=\"upper right\")\n"
            "ax.grid(True, alpha=0.3, which=\"both\")\n"
            "fig.tight_layout()\n"
            "out_c = PLOTS_DIR / \"training_compute_vs_mae.png\"\n"
            "fig.savefig(out_c, dpi=150, bbox_inches=\"tight\")\n"
            "plt.show()\n"
            "print(f\"Wrote {out_c}\")"
        ),
    ],
}


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "notebooks" / "thesis_figures.ipynb"
    out.write_text(json.dumps(NB, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out} ({len(NB['cells'])} cells)")


if __name__ == "__main__":
    main()
