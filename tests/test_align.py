import json
import tempfile

import numpy as np
import pytest

from src.align import (
    resample_to_length,
    align_lengths,
    build_muscle_mapping,
    build_mapping_by_shared_names,
    build_mapping_kinesis_artifacts,
)


def test_resample_identity():
    arr = np.random.rand(10, 3).astype(np.float32)
    out = resample_to_length(arr, 10)
    np.testing.assert_allclose(out, arr)


def test_align_lengths_resample():
    a = np.ones((10, 4), dtype=np.float32)
    b = np.ones((15, 4), dtype=np.float32)
    a2, b2 = align_lengths(a, b, resample=True)
    assert a2.shape == b2.shape


def test_align_lengths_feature_mismatch():
    a = np.ones((10, 4), dtype=np.float32)
    b = np.ones((10, 5), dtype=np.float32)
    with pytest.raises(ValueError):
        align_lengths(a, b)


def test_build_muscle_mapping():
    payload = {"mapping": [
        {"rajagopal": "a", "kinesis": "x"},
        {"rajagopal": "b", "kinesis": None},
        {"rajagopal": "c", "kinesis": "z"},
    ]}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(payload, f)
        path = f.name
    mapping = build_muscle_mapping(path, ["a", "b", "c"], ["x", "y", "z"])
    assert len(mapping) == 2
    assert mapping.shared_names == ["a", "c"]


def test_build_mapping_by_shared_names() -> None:
    raj = ["m1", "m2", "m3"]
    kin = ["m2", "m3", "m4"]
    mapping = build_mapping_by_shared_names(raj, kin)
    assert len(mapping) == 2
    assert mapping.shared_names == ["m2", "m3"]
    assert mapping.rajagopal_indices == [1, 2]
    assert mapping.kinesis_indices == [0, 1]


def test_build_mapping_kinesis_artifacts() -> None:
    raj = ["glmax1_r", "semimem_r", "bflh_r"]
    kin = ["glut_max1_r", "semimem_r", "bifemlh_r"]
    alias = {"glut_max1_r": "glmax1_r", "bifemlh_r": "bflh_r", "semimem_r": "semimem_r"}
    mapping = build_mapping_kinesis_artifacts(raj, kin, alias)
    assert len(mapping) == 3
    assert mapping.shared_names == ["glmax1_r", "semimem_r", "bflh_r"]
