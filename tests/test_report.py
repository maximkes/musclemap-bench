import csv
from pathlib import Path

from src import report


def _fixture_results(*, with_layer2: bool = False) -> dict:
    layer1_mm = {
        "n_samples": 2,
        "mae": 0.1,
        "rmse": 0.2,
        "dtw_mae": 0.11,
        "dtw_rmse": 0.21,
        "pearson_r_mean": 0.5,
        "r2_mean": 0.4,
        "onset_timing_error_mean": 2.0,
        "energy_ratio": 1.0,
        "coactivation_frobenius": 0.3,
        "r2_per_muscle_named": {"m1": 0.5, "m2": 0.3},
        "onset_timing_errors_all": [1.0, 3.0, 2.0],
    }
    layer1_kin = {"n_samples": 2, "mae": 0.15, "r2_mean": 0.2, "r2_per_muscle_named": {"m1": 0.2}}
    layer2 = {}
    if with_layer2:
        layer2 = {
            "musclemap": {"fid_mean": 1.2, "r_precision_top1": 0.4, "mm_dist": 2.0, "diversity": 3.0},
            "motiongpt": {"fid_mean": 1.5, "r_precision_top1": 0.3, "mm_dist": 2.5, "diversity": 2.8},
        }
    layer1_paired = {
        "n_sequences": 2,
        "n_muscles": 2,
        "musclemap": {"n_samples": 2, "mae": 0.08, "rmse": 0.12, "r2_per_muscle_named": {"m1": 0.4, "m2": 0.2}},
        "kinesis": {"n_samples": 2, "mae": 0.05, "rmse": 0.09, "r2_per_muscle_named": {"m1": 0.3, "m2": 0.1}},
    }
    return {
        "meta": {"n_samples": 2, "n_paired_sequences": 2, "device": "cpu"},
        "layer1": {"musclemap": layer1_mm, "kinesis": layer1_kin},
        "layer1_paired": layer1_paired,
        "layer2": layer2,
        "resources": {
            "musclemap": {"inference": {"mean_s": 0.5}, "training_gpu_hours": 10},
            "kinesis": {"inference": {"mean_s": 0.2}},
            "motiongpt": {"inference": {"mean_s": 1.0}},
        },
    }


def test_layer1_paired_available() -> None:
    assert report.layer1_paired_available(_fixture_results())
    empty = _fixture_results()
    empty["layer1_paired"] = {}
    assert not report.layer1_paired_available(empty)


def test_layer2_available_false() -> None:
    assert not report.layer2_available(_fixture_results(with_layer2=False))


def test_layer2_available_true() -> None:
    assert report.layer2_available(_fixture_results(with_layer2=True))


def test_generate_all_skips_layer2(tmp_path: Path) -> None:
    results = _fixture_results(with_layer2=False)
    report.generate_all(results, tmp_path)
    assert (tmp_path / "table_layer1.tex").is_file()
    assert (tmp_path / "layer2_skipped.txt").is_file()
    assert not (tmp_path / "table_layer2.tex").exists()
    assert (tmp_path / "table_layer1_paired.tex").is_file()
    assert (tmp_path / "plots" / "per_muscle_r2_paired.png").is_file()
    assert (tmp_path / "summary_metrics.csv").is_file()
    csv_text = (tmp_path / "summary_metrics.csv").read_text(encoding="utf-8")
    assert "skipped" in csv_text
    assert "layer1_paired" in csv_text


def test_generate_all_with_layer2(tmp_path: Path) -> None:
    results = _fixture_results(with_layer2=True)
    report.generate_all(results, tmp_path)
    tex = (tmp_path / "table_layer2.tex").read_text(encoding="utf-8")
    assert r"\begin{tabular}" in tex
    assert "FID" in tex
    assert (tmp_path / "plots" / "per_muscle_r2.caption.txt").is_file()


def test_write_csv_roundtrip(tmp_path: Path) -> None:
    report.write_csv_summary(_fixture_results(with_layer2=True), tmp_path / "out.csv")
    with (tmp_path / "out.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["layer", "method", "metric", "value"]
    assert any(r[0] == "layer2" for r in rows[1:])
