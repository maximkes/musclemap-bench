from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_EPOCH_STEM_RE = re.compile(r"epoch_(\d+)$", re.IGNORECASE)


def discover_checkpoints(checkpoints_dir: str | Path) -> list[Path]:
    """Return checkpoint paths sorted by epoch number (``epoch_XXXX.pt``)."""
    root = Path(checkpoints_dir)
    if not root.is_dir():
        return []
    paths = sorted(root.glob("epoch_*.pt"), key=_checkpoint_sort_key)
    return paths


def checkpoint_stem(path: str | Path) -> str:
    """Return stem like ``epoch_0004`` from a checkpoint path."""
    return Path(path).stem


def parse_epoch(path: str | Path) -> int:
    """Parse epoch index from ``epoch_XXXX`` stem."""
    stem = checkpoint_stem(path)
    m = _EPOCH_STEM_RE.match(stem)
    if not m:
        raise ValueError(f"Cannot parse epoch from checkpoint stem: {stem}")
    return int(m.group(1))


def _checkpoint_sort_key(path: Path) -> int:
    try:
        return parse_epoch(path)
    except ValueError:
        return 10**9


def resolve_steps_per_epoch(train_cfg: dict[str, Any], curve_cfg: dict[str, Any] | None = None) -> int:
    """Micro-batches per training epoch (``len(train_loader)``)."""
    curve_cfg = curve_cfg or {}
    if curve_cfg.get("steps_per_epoch") is not None:
        return max(1, int(curve_cfg["steps_per_epoch"]))

    training = train_cfg.get("training", {})
    batch_size = int(curve_cfg.get("batch_size", training.get("batch_size", 1)))
    train_sequences = curve_cfg.get("train_sequences")
    if train_sequences is not None:
        return max(1, math.ceil(int(train_sequences) / max(1, batch_size)))

    raise ValueError(
        "steps_per_epoch is unknown: set training_curve.steps_per_epoch or "
        "training_curve.train_sequences in config.yaml"
    )


def optimizer_steps_per_epoch(steps_per_epoch: int, accumulation_steps: int) -> int:
    """Optimizer steps per epoch (matches MuscleMAPTrainer micro-batch schedule)."""
    steps_per_epoch = max(1, int(steps_per_epoch))
    accum = max(1, int(accumulation_steps))
    return math.ceil(steps_per_epoch / accum)


def epoch_to_global_step(
    epoch: int,
    train_cfg: dict[str, Any],
    *,
    curve_cfg: dict[str, Any] | None = None,
) -> int:
    """Global optimizer step after completing training epoch ``epoch`` (0-based)."""
    curve_cfg = curve_cfg or {}
    steps_per_epoch = resolve_steps_per_epoch(train_cfg, curve_cfg)
    accum = int(
        curve_cfg.get(
            "accumulation_steps",
            train_cfg.get("training", {}).get("accumulation_steps", 1),
        )
    )
    opt_steps = optimizer_steps_per_epoch(steps_per_epoch, accum)
    return (int(epoch) + 1) * opt_steps


def load_val_metrics(
    results_dir: str | Path,
    stems: list[str] | None = None,
    *,
    split: str = "val",
) -> pd.DataFrame:
    """Load validation metrics JSON files into a DataFrame."""
    root = Path(results_dir)
    if stems is None:
        stems = sorted(p.stem.replace(f"{split}_", "", 1) for p in root.glob(f"{split}_epoch_*_metrics.json"))

    rows: list[dict[str, Any]] = []
    for stem in stems:
        path = root / f"{split}_{stem}_metrics.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        epoch = parse_epoch(stem) if stem.startswith("epoch_") else None
        rows.append(
            {
                "stem": stem,
                "epoch": epoch if epoch is not None else int(stem.split("_")[-1]),
                "mae": float(data["mae"]),
                "rmse": float(data["rmse"]),
                "mpjae": float(data.get("mpjae", data["mae"])),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["stem", "epoch", "mae", "rmse", "mpjae", "global_step"])
    df = pd.DataFrame(rows).sort_values("epoch").reset_index(drop=True)
    return df


def attach_global_steps(
    df: pd.DataFrame,
    train_cfg: dict[str, Any],
    *,
    curve_cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Add ``global_step`` column from epoch indices."""
    out = df.copy()
    out["global_step"] = [
        epoch_to_global_step(int(e), train_cfg, curve_cfg=curve_cfg) for e in out["epoch"]
    ]
    return out


def interpolate_curve(
    df: pd.DataFrame,
    step_grid: np.ndarray | list[int],
    *,
    value_col: str,
    x_col: str = "global_step",
) -> np.ndarray:
    """Linearly interpolate ``value_col`` over ``step_grid`` (numpy.interp)."""
    if df.empty or len(df) < 2:
        if df.empty:
            return np.full(len(step_grid), np.nan, dtype=np.float64)
        return np.full(len(step_grid), float(df[value_col].iloc[0]), dtype=np.float64)

    x = df[x_col].to_numpy(dtype=np.float64)
    y = df[value_col].to_numpy(dtype=np.float64)
    order = np.argsort(x)
    x, y = x[order], y[order]
    grid = np.asarray(step_grid, dtype=np.float64)
    return np.interp(grid, x, y, left=y[0], right=y[-1])


def build_val_metrics_table(
    checkpoints: list[Path],
    results_dir: Path,
    train_cfg: dict[str, Any],
    *,
    curve_cfg: dict[str, Any] | None = None,
    split: str = "val",
) -> pd.DataFrame:
    """Merge discovered checkpoints with on-disk val metrics and global steps."""
    stems = [checkpoint_stem(p) for p in checkpoints]
    df = load_val_metrics(results_dir, stems=stems, split=split)
    if df.empty:
        return df
    return attach_global_steps(df, train_cfg, curve_cfg=curve_cfg)
