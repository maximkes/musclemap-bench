import numpy as np
import pytest
import torch

from src import metrics_l2
from src.metrics_l2 import (
    _T2MEvaluator,
    _encode_motion_batch,
    _encode_text_batch,
    _stub_word_vectorizer,
    compute_fid,
    compute_l2_metrics,
    extract_motion_features,
    extract_text_features,
    reset_evaluator_cache,
)


_EMBED_DIM = 32


class _FakeTextEncoder(torch.nn.Module):
    def forward(self, word_embs: torch.Tensor, pos_ohot: torch.Tensor, text_lengths: torch.Tensor) -> torch.Tensor:
        _ = (word_embs, pos_ohot, text_lengths)
        batch = int(pos_ohot.shape[0])
        return torch.zeros(batch, 1, _EMBED_DIM)


class _FakeMoveEncoder(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, steps, _ = inputs.shape
        return torch.zeros(batch, steps, _EMBED_DIM)


class _FakeMotionEncoder(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, m_lens: torch.Tensor) -> torch.Tensor:
        _ = (inputs, m_lens)
        batch = int(m_lens.shape[0])
        return torch.zeros(batch, _EMBED_DIM)


@pytest.fixture(autouse=True)
def _clear_evaluator_cache() -> None:
    reset_evaluator_cache()
    yield
    reset_evaluator_cache()


@pytest.fixture()
def fake_evaluator() -> _T2MEvaluator:
    mean = np.zeros(263, dtype=np.float32)
    std = np.ones(263, dtype=np.float32)
    return _T2MEvaluator(
        t2m_textencoder=_FakeTextEncoder(),
        t2m_moveencoder=_FakeMoveEncoder(),
        t2m_motionencoder=_FakeMotionEncoder(),
        w_vectorizer=_stub_word_vectorizer(),
        mean=mean,
        std=std,
        unit_len=4,
        max_text_len=20,
        device="cpu",
    )


@pytest.fixture()
def bench_cfg() -> dict:
    return {
        "paths": {"musclemap_model_repo": "../musclemap-model", "musclemap_train_config": "../musclemap-model/config/train.yaml"},
        "inference": {"device": "cpu"},
        "layer2": {"n_replication": 2, "top_k": [1, 2]},
    }


def test_compute_fid_identical():
    feats = np.random.default_rng(0).normal(size=(32, 16)).astype(np.float32)
    assert compute_fid(feats, feats.copy()) == pytest.approx(0.0, abs=1e-4)


def test_encode_text_batch_shape(fake_evaluator: _T2MEvaluator) -> None:
    texts = ["walk forward", "jump high"]
    out = _encode_text_batch(texts, fake_evaluator)
    assert out.shape == (2, _EMBED_DIM)
    assert out.dtype == np.float32


def test_encode_motion_batch_shape(fake_evaluator: _T2MEvaluator) -> None:
    motions = [
        np.ones((40, 263), dtype=np.float32),
        np.ones((24, 263), dtype=np.float32) * 2.0,
    ]
    out = _encode_motion_batch(motions, fake_evaluator)
    assert out.shape == (2, _EMBED_DIM)
    assert out.dtype == np.float32


def test_extract_features_with_mock(fake_evaluator: _T2MEvaluator, bench_cfg: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metrics_l2, "_load_evaluator", lambda _cfg: fake_evaluator)
    motions = [np.zeros((32, 263), dtype=np.float32)]
    texts = ["a person walks"]
    m_feats = extract_motion_features(motions, bench_cfg)
    t_feats = extract_text_features(texts, bench_cfg)
    assert m_feats.shape[0] == 1
    assert t_feats.shape[0] == 1


def test_compute_l2_metrics_with_mock(fake_evaluator: _T2MEvaluator, bench_cfg: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metrics_l2, "_load_evaluator", lambda _cfg: fake_evaluator)
    rng = np.random.default_rng(1)
    motions = [rng.normal(size=(48, 263)).astype(np.float32) for _ in range(6)]
    texts = [f"action {i}" for i in range(6)]
    out = compute_l2_metrics(motions, texts, bench_cfg)
    assert "fid_mean" in out
    assert "r_precision_top1" in out
    assert "mm_dist" in out
    assert "diversity" in out


def test_motion_feature_dim_guard(fake_evaluator: _T2MEvaluator, bench_cfg: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metrics_l2, "_load_evaluator", lambda _cfg: fake_evaluator)
    with pytest.raises(ValueError, match="263"):
        extract_motion_features([np.ones((10, 100), dtype=np.float32)], bench_cfg)
