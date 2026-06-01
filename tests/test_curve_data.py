import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.curve_data import (
    discover_checkpoints,
    epoch_to_global_step,
    interpolate_curve,
    load_val_metrics,
    optimizer_steps_per_epoch,
    parse_epoch,
)


def test_optimizer_steps_per_epoch():
    assert optimizer_steps_per_epoch(5416, 16) == 339


def test_epoch_to_global_step_datasphere():
    train_cfg = {"training": {"batch_size": 16, "accumulation_steps": 2}}
    curve_cfg = {"steps_per_epoch": 5416, "accumulation_steps": 16}
    assert epoch_to_global_step(3, train_cfg, curve_cfg=curve_cfg) == 4 * 339


def test_interpolate_curve():
    df = pd.DataFrame({"global_step": [100, 300], "mae": [0.2, 0.1]})
    y = interpolate_curve(df, [100, 200, 300], value_col="mae")
    np.testing.assert_allclose(y, [0.2, 0.15, 0.1])


def test_load_val_metrics(tmp_path: Path):
    payload = {"mae": 0.1, "rmse": 0.2}
    (tmp_path / "val_epoch_0004_metrics.json").write_text(json.dumps(payload), encoding="utf-8")
    df = load_val_metrics(tmp_path, stems=["epoch_0004"])
    assert len(df) == 1
    assert df.iloc[0]["epoch"] == 4
    assert df.iloc[0]["mae"] == pytest.approx(0.1)


def test_discover_checkpoints_sorted(tmp_path: Path):
    for ep in (24, 4, 9):
        (tmp_path / f"epoch_{ep:04d}.pt").write_bytes(b"x")
    paths = discover_checkpoints(tmp_path)
    assert [parse_epoch(p) for p in paths] == [4, 9, 24]
