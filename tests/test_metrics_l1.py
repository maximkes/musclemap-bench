import numpy as np
import pytest

from src.metrics_l1 import compute_l1_metrics, aggregate_l1_metrics


@pytest.fixture()
def rng():
    return np.random.default_rng(0)


def test_perfect_pair(rng):
    arr = rng.uniform(0, 1, (100, 10)).astype(np.float32)
    names = [f"m{i}" for i in range(10)]
    out = compute_l1_metrics(arr, arr.copy(), names, dtw_enabled=False, compute_coactivation=True)
    assert out["mae"] == pytest.approx(0.0, abs=1e-7)
    assert out["rmse"] == pytest.approx(0.0, abs=1e-7)
    assert out["r2_mean"] == pytest.approx(1.0, abs=1e-4)
    assert out["coactivation_frobenius"] == pytest.approx(0.0, abs=1e-4)


def test_noisy_pair(rng):
    ref = rng.uniform(0, 1, (80, 10)).astype(np.float32)
    pred = np.clip(ref + rng.normal(0, 0.1, ref.shape), 0, 1).astype(np.float32)
    names = [f"m{i}" for i in range(10)]
    out = compute_l1_metrics(pred, ref, names, dtw_enabled=False, compute_coactivation=False)
    assert out["mae"] > 0
    assert out["rmse"] > 0
    assert "energy_pred" in out and "energy_ref" in out


def test_dtw_not_worse_for_shifted_signal(rng):
    ref = rng.uniform(0, 1, (120, 6)).astype(np.float32)
    pred = np.roll(ref, shift=5, axis=0)
    names = [f"m{i}" for i in range(6)]
    raw = compute_l1_metrics(pred, ref, names, dtw_enabled=False, compute_coactivation=False)
    dtw = compute_l1_metrics(pred, ref, names, dtw_enabled=True, dtw_radius=10, compute_coactivation=False)
    assert dtw["dtw_mae"] <= raw["mae"] + 1e-3


def test_aggregate(rng):
    names = [f"m{i}" for i in range(4)]
    items = []
    for _ in range(3):
        ref = rng.uniform(0, 1, (50, 4)).astype(np.float32)
        pred = np.clip(ref + rng.normal(0, 0.05, ref.shape), 0, 1).astype(np.float32)
        items.append(compute_l1_metrics(pred, ref, names, dtw_enabled=False, compute_coactivation=False))
    agg = aggregate_l1_metrics(items)
    assert agg["n_samples"] == 3
    assert "r2_per_muscle_named" in agg


def test_shape_mismatch_raises(rng):
    pred = rng.uniform(0, 1, (40, 5)).astype(np.float32)
    ref = rng.uniform(0, 1, (40, 6)).astype(np.float32)
    with pytest.raises(ValueError):
        compute_l1_metrics(pred, ref, [f"m{i}" for i in range(5)], dtw_enabled=False)
