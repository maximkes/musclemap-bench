import numpy as np
import pytest

from src.loaders import BenchmarkSample, get_benchmark_sample, normalize_dataset_sample


def test_normalize_dict_acts_mask():
    acts = np.arange(20, dtype=np.float32).reshape(4, 5)
    mask = np.array([True, True, False, False])
    raw = {"text": "walk forward", "acts": acts, "mask": mask, "motion": np.zeros((4, 322), np.float32)}
    sample = normalize_dataset_sample(raw, sequence_id="seq_a")
    assert sample.sequence_id == "seq_a"
    assert sample.text == "walk forward"
    assert sample.activations.shape == (2, 5)
    assert sample.motion is not None and sample.motion.shape == (2, 322)


def test_normalize_dict_activations_key():
    acts = np.ones((3, 80), dtype=np.float32)
    sample = normalize_dataset_sample({"text": "jump", "activations": acts}, sequence_id="s1")
    np.testing.assert_array_equal(sample.activations, acts)
    assert sample.motion is None


def test_normalize_dict_acts_true_t():
    acts = np.ones((6, 4), dtype=np.float32)
    sample = normalize_dataset_sample({"text": "run", "acts": acts, "true_T": 3}, sequence_id="s2")
    assert sample.activations.shape == (3, 4)


def test_normalize_tuple_five_fields():
    acts = np.ones((5, 80), dtype=np.float32)
    mask = np.array([1, 1, 1, 0, 0], dtype=bool)
    motion = np.zeros((5, 322), dtype=np.float32)
    raw = ("squat", motion, acts, mask, 3)
    sample = normalize_dataset_sample(raw, sequence_id="tuple5")
    assert sample.activations.shape == (3, 80)
    assert sample.motion.shape == (3, 322)


def test_normalize_tuple_text_acts():
    acts = np.ones((2, 10), dtype=np.float32)
    sample = normalize_dataset_sample(("wave", acts), sequence_id="tuple2")
    np.testing.assert_array_equal(sample.activations, acts)


def test_normalize_tuple_acts_text():
    acts = np.ones((2, 10), dtype=np.float32)
    sample = normalize_dataset_sample((acts, "turn"), sequence_id="tuple2b")
    assert sample.text == "turn"


def test_get_benchmark_sample_uses_items():
    class _FakeDS:
        _items = [(__import__("pathlib").Path("/data/my_seq_dir"), 0, 2, "walk")]

        def __getitem__(self, idx: int) -> dict:
            acts = np.ones((4, 80), dtype=np.float32)
            return {
                "text": "walk",
                "acts": acts,
                "mask": np.array([1, 1, 0, 0], dtype=bool),
            }

    sample = get_benchmark_sample(_FakeDS(), 0)
    assert sample.sequence_id == "my_seq_dir"
    assert sample.activations.shape == (2, 80)


def test_missing_acts_raises():
    with pytest.raises(KeyError):
        normalize_dataset_sample({"text": "x"}, sequence_id="bad")
