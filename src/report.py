from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

LAYER1_METRICS: tuple[tuple[str, str, int, bool], ...] = (
    ("MAE (raw)", "mae", 3, False),
    ("MAE (DTW)", "dtw_mae", 3, False),
    ("RMSE (raw)", "rmse", 3, False),
    ("RMSE (DTW)", "dtw_rmse", 3, False),
    ("Pearson-r", "pearson_r_mean", 3, False),
    ("R^2", "r2_mean", 3, False),
    ("Onset error (frames)", "onset_timing_error_mean", 1, False),
    ("Energy ratio", "energy_ratio", 3, False),
    ("Co-activation Frobenius", "coactivation_frobenius", 3, False),
)

LAYER2_METRICS: tuple[tuple[str, str, int, bool], ...] = (
    ("FID", "fid_mean", 2, False),
    ("R-Precision Top-1", "r_precision_top1", 3, True),
    ("R-Precision Top-2", "r_precision_top2", 3, True),
    ("R-Precision Top-3", "r_precision_top3", 3, True),
    ("MM-Dist", "mm_dist", 3, False),
    ("Diversity", "diversity", 3, False),
)

CAPTION_LAYER1_R2 = (
    "Layer 1 — per-muscle coefficient of determination (R²) on the held-out test split. "
    "Bars compare MuscleMAP text-to-activation predictions against Kinesis/MyoLeg precomputed "
    "activations on the shared Rajagopal muscle subset."
)
CAPTION_LAYER1_PAIRED = (
    "Layer 1 paired comparison: only sequences with successful Kinesis precompute artifacts. "
    "MuscleMAP and Kinesis are evaluated on the same clips and the same mapped leg-muscle subset."
)
CAPTION_LAYER1_ONSET = (
    "Layer 1 — distribution of activation onset timing errors (frames) for MuscleMAP. "
    "Onset is the first frame where activation stays above the configured threshold "
    "for at least min_active_frames consecutive frames."
)
CAPTION_LAYER2_UNAVAILABLE = (
    "Layer 2 (HumanML3D text-to-motion metrics) was not computed for this run. "
    "Re-run the benchmark with layer2.enabled and without --skip-layer2 once MotionGPT "
    "checkpoints and evaluator assets are available."
)


def _fmt(val: Any, decimals: int = 3, percent: bool = False) -> str:
    if val is None:
        return "--"
    f = float(val)
    return f"{f * 100:.1f}" if percent else f"{f:.{decimals}f}"


def _branch_has_metrics(branch: dict[str, Any], metric_rows: tuple[tuple[str, str, int, bool], ...]) -> bool:
    if not branch:
        return False
    return any(branch.get(key) is not None for _, key, _, _ in metric_rows)


def layer1_paired_available(results: dict[str, Any]) -> bool:
    """Return True when paired Layer 1 aggregates exist."""
    paired = results.get("layer1_paired", {})
    if paired.get("error"):
        return False
    return _branch_has_metrics(paired.get("musclemap", {}), LAYER1_METRICS)


def layer2_available(results: dict[str, Any]) -> bool:
    """Return True when Layer 2 aggregates contain at least one metric."""
    layer2 = results.get("layer2", {})
    return _branch_has_metrics(layer2.get("musclemap", {}), LAYER2_METRICS) or _branch_has_metrics(
        layer2.get("motiongpt", {}), LAYER2_METRICS
    )


def _write_caption(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def write_table_layer1(results: dict[str, Any], out_path: Path) -> None:
    mm = results["layer1"].get("musclemap", {})
    kin = results["layer1"].get("kinesis", {})
    n_mm = mm.get("n_samples", results.get("meta", {}).get("n_samples", "?"))
    caption = (
        f"Layer 1 aggregate metrics over {n_mm} test sequences. "
        "MuscleMAP: full Rajagopal set; Kinesis: mapped subset from precomputed artifacts."
    )
    lines = [
        r"% " + caption,
        r"\begin{tabular}{lcc}",
        r"\caption{" + caption + r"}",
        r"\label{tab:layer1}",
        r"\toprule",
        r"Metric & MuscleMAP & Kinesis \\",
        r"\midrule",
    ]
    for label, key, dec, pct in LAYER1_METRICS:
        lines.append(f"{label} & {_fmt(mm.get(key), dec, pct)} & {_fmt(kin.get(key), dec, pct)} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_table_layer1_paired(results: dict[str, Any], out_path: Path) -> None:
    if not layer1_paired_available(results):
        return
    paired = results["layer1_paired"]
    mm = paired.get("musclemap", {})
    kin = paired.get("kinesis", {})
    n_seq = paired.get("n_sequences", mm.get("n_samples", "?"))
    n_muscles = paired.get("n_muscles", "?")
    caption = (
        f"Paired Layer 1 metrics over {n_seq} sequences and {n_muscles} mapped leg muscles. "
        "MuscleMAP and Kinesis use the same clips and muscle subset."
    )
    lines = [
        r"% " + caption,
        r"\begin{tabular}{lcc}",
        r"\caption{" + caption + r"}",
        r"\label{tab:layer1_paired}",
        r"\toprule",
        r"Metric & MuscleMAP & Kinesis \\",
        r"\midrule",
    ]
    for label, key, dec, pct in LAYER1_METRICS:
        lines.append(f"{label} & {_fmt(mm.get(key), dec, pct)} & {_fmt(kin.get(key), dec, pct)} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_table_layer2(results: dict[str, Any], out_path: Path) -> None:
    if not layer2_available(results):
        out_path.write_text(
            "% Layer 2 not available for this run.\n"
            + CAPTION_LAYER2_UNAVAILABLE
            + "\n",
            encoding="utf-8",
        )
        return
    mm = results["layer2"].get("musclemap", {})
    mg = results["layer2"].get("motiongpt", {})
    caption = (
        "Layer 2 HumanML3D-style metrics (lower FID / MM-Dist is better; higher R-precision "
        "and diversity are better). Compares generated motions from MuscleMAP and MotionGPT."
    )
    lines = [
        r"% " + caption,
        r"\begin{tabular}{lcc}",
        r"\caption{" + caption + r"}",
        r"\label{tab:layer2}",
        r"\toprule",
        r"Metric & MuscleMAP & MotionGPT \\",
        r"\midrule",
    ]
    for label, key, dec, pct in LAYER2_METRICS:
        lines.append(f"{label} & {_fmt(mm.get(key), dec, pct)} & {_fmt(mg.get(key), dec, pct)} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_table_resources(results: dict[str, Any], out_path: Path) -> None:
    res = results["resources"]
    mm = res["musclemap"]
    kin = res["kinesis"]
    mg = res["motiongpt"]
    caption = "Training and inference resource summary per method."

    def t(x: dict[str, Any]) -> str:
        v = x["inference"].get("mean_s")
        return "--" if v is None else f"{v:.3f}s"

    def gh(x: dict[str, Any]) -> str:
        v = x.get("training_gpu_hours")
        if v is None:
            return "--"
        return f"{float(v):.0f}"

    lines = [
        r"% " + caption,
        r"\begin{tabular}{lccc}",
        r"\caption{" + caption + r"}",
        r"\label{tab:resources}",
        r"\toprule",
        r" & MuscleMAP & Kinesis & MotionGPT \\",
        r"\midrule",
        f"Training GPU-hours & {gh(mm)} & {gh(kin)} & {gh(mg)} \\",
        f"Training GPU type & {mm.get('training_gpu_type', '--')} & {kin.get('training_gpu_type', '--')} & {mg.get('training_gpu_type', '--')} \\",
        f"Inference / sample & {t(mm)} & {t(kin)} & {t(mg)} \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_csv_summary(results: dict[str, Any], out_path: Path) -> None:
    """Export scalar metrics as a wide CSV (layer, method, metric, value)."""
    rows: list[tuple[str, str, str, str]] = []
    meta = results.get("meta", {})
    if meta:
        for key, val in sorted(meta.items()):
            rows.append(("meta", "benchmark", key, str(val)))

    for label, key, _, _ in LAYER1_METRICS:
        for method, branch in results.get("layer1", {}).items():
            if branch.get(key) is not None:
                rows.append(("layer1", method, label, _fmt(branch.get(key), 6, False)))

    if layer1_paired_available(results):
        paired = results["layer1_paired"]
        for label, key, _, _ in LAYER1_METRICS:
            for method in ("musclemap", "kinesis"):
                branch = paired.get(method, {})
                if branch.get(key) is not None:
                    rows.append(("layer1_paired", method, label, _fmt(branch.get(key), 6, False)))

    if layer2_available(results):
        for label, key, _, _ in LAYER2_METRICS:
            for method, branch in results.get("layer2", {}).items():
                if branch.get(key) is not None:
                    rows.append(("layer2", method, label, _fmt(branch.get(key), 6, False)))
    else:
        rows.append(("layer2", "—", "status", "skipped"))

    res = results.get("resources", {})
    for method, branch in res.items():
        inf = branch.get("inference", {})
        if inf.get("mean_s") is not None:
            rows.append(("resources", method, "inference_mean_s", f"{float(inf['mean_s']):.6f}"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["layer", "method", "metric", "value"])
        writer.writerows(rows)


def plot_per_muscle_r2_paired(results: dict[str, Any], out_path: Path) -> None:
    if not layer1_paired_available(results):
        return
    paired = results["layer1_paired"]
    mm_named = paired.get("musclemap", {}).get("r2_per_muscle_named", {})
    kin_named = paired.get("kinesis", {}).get("r2_per_muscle_named", {})
    if not mm_named:
        return
    names = sorted(set(mm_named) & set(kin_named)) if kin_named else sorted(mm_named)
    mm_vals = [mm_named[n] for n in names]
    kin_vals = [kin_named.get(n, np.nan) for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(12, 0.35 * len(names)), 5))
    ax.bar(x - 0.2, mm_vals, width=0.4, label="MuscleMAP")
    ax.bar(x + 0.2, np.nan_to_num(kin_vals, nan=0.0), width=0.4, label="Kinesis")
    ax.axhline(0.0, color="gray", linewidth=0.5)
    ax.set_ylabel("R² per muscle")
    n_seq = paired.get("n_sequences", "?")
    ax.set_title(f"Layer 1 paired: per-muscle R² (n={n_seq} sequences)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    _write_caption(out_path.with_suffix(".caption.txt"), CAPTION_LAYER1_PAIRED)


def plot_per_muscle_r2(results: dict[str, Any], out_path: Path) -> None:
    mm_named = results["layer1"].get("musclemap", {}).get("r2_per_muscle_named", {})
    if not mm_named:
        return
    kin_named = results["layer1"].get("kinesis", {}).get("r2_per_muscle_named", {})
    names = list(mm_named.keys())
    mm_vals = [mm_named[n] for n in names]
    kin_vals = [kin_named.get(n, np.nan) for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(12, 0.35 * len(names)), 5))
    ax.bar(x - 0.2, mm_vals, width=0.4, label="MuscleMAP")
    if np.isfinite(np.asarray(kin_vals, dtype=float)).any():
        ax.bar(x + 0.2, np.nan_to_num(kin_vals, nan=0.0), width=0.4, label="Kinesis (precompute)")
    ax.set_ylabel("R² per muscle")
    ax.set_xlabel("Rajagopal muscle (shared subset for Kinesis)")
    title = "Layer 1: Per-muscle R² — MuscleMAP vs Kinesis"
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.legend(loc="upper right", fontsize=8)
    ax.axhline(0.0, color="gray", linewidth=0.5)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    _write_caption(out_path.with_suffix(".caption.txt"), CAPTION_LAYER1_R2)


def plot_onset_timing(results: dict[str, Any], out_path: Path) -> None:
    vals = results["layer1"].get("musclemap", {}).get("onset_timing_errors_all", [])
    if not vals:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(vals, bins=min(20, max(5, len(vals) // 5)), edgecolor="white")
    ax.set_xlabel("Onset timing error (frames)")
    ax.set_ylabel("Count (muscle × sample)")
    ax.set_title("Layer 1: MuscleMAP activation onset timing error")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    _write_caption(out_path.with_suffix(".caption.txt"), CAPTION_LAYER1_ONSET)


def generate_all(
    results: dict[str, Any],
    results_dir: Path,
    *,
    export_csv: bool = True,
) -> None:
    """Write LaTeX tables, plots, captions, and optional CSV summary."""
    results_dir = Path(results_dir)
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    write_table_layer1(results, results_dir / "table_layer1.tex")
    write_table_layer1_paired(results, results_dir / "table_layer1_paired.tex")
    write_table_resources(results, results_dir / "table_resources.tex")
    plot_per_muscle_r2(results, plots_dir / "per_muscle_r2.png")
    plot_per_muscle_r2_paired(results, plots_dir / "per_muscle_r2_paired.png")
    plot_onset_timing(results, plots_dir / "onset_timing.png")

    if layer2_available(results):
        write_table_layer2(results, results_dir / "table_layer2.tex")
    else:
        logger.info("Layer 2 metrics absent; skipping MotionGPT comparison table.")
        note_path = results_dir / "layer2_skipped.txt"
        note_path.write_text(CAPTION_LAYER2_UNAVAILABLE + "\n", encoding="utf-8")
        skipped = results_dir / "table_layer2.tex"
        if skipped.exists():
            skipped.unlink()

    if export_csv:
        write_csv_summary(results, results_dir / "summary_metrics.csv")
