import numpy as np
import torch

from src.inference import _motion_from_output, _predicted_length, run_motiongpt, run_musclemap


class _StubBackbone:
    def __init__(self) -> None:
        self._musclemap_bench_eval_patch = False

    def forward(self, batch: dict, task: str = "t2m") -> dict:
        return {}


class _StubMuscleMAP(torch.nn.Module):
    config = {"model": {"length_predictor": {"min_T": 2, "max_T": 10}}}

    def __init__(self) -> None:
        super().__init__()
        self.backbone = _StubBackbone()

    def forward(
        self,
        text_tokens: list[str],
        motion_tokens=None,
        lengths=None,
        T_frame=None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        _ = (text_tokens, motion_tokens, lengths, T_frame)
        logits = torch.full((1, 6, 80), 2.0)
        pred_log_T = torch.log(torch.tensor([4.0]))
        return logits, pred_log_T, {"feats": torch.ones(1, 5, 263), "length": [4]}


def test_motion_from_output_feats_length():
    out = {"feats": torch.arange(15, dtype=torch.float32).reshape(1, 5, 3), "length": [3]}
    motion = _motion_from_output(out)
    assert motion is not None
    assert motion.shape == (3, 3)
    assert motion.dtype == np.float32


def test_motion_from_output_empty_dict():
    assert _motion_from_output({}) is None


def test_predicted_length_clamps_to_logits():
    logits = torch.zeros(1, 6, 80)
    pred_log_t = torch.log(torch.tensor([4.0]))
    assert _predicted_length(logits, pred_log_t, _StubMuscleMAP.config) == 4


def test_run_musclemap_sigmoid_and_trim():
    model = _StubMuscleMAP()
    sample = run_musclemap(model, "walk", "seq1", ref_T=3)
    assert sample.sequence_id == "seq1"
    assert sample.activations is not None
    assert sample.activations.shape == (3, 80)
    assert np.all(sample.activations > 0.5)
    assert sample.motion is not None
    assert sample.motion.shape == (3, 263)
    assert sample.meta["pred_T"] == 4
    assert getattr(model.backbone, "_musclemap_bench_eval_patch", False)


def test_run_musclemap_patches_backbone_once():
    model = _StubMuscleMAP()
    run_musclemap(model, "a", "s0")
    first_forward = model.backbone.forward
    run_musclemap(model, "b", "s1")
    assert model.backbone.forward is first_forward


class _StubMotionGPT(torch.nn.Module):
    def forward(self, batch: dict, task: str = "t2m") -> dict:
        assert task == "t2m"
        assert batch["text"] == ["jump high"]
        assert batch["length"] == [196]
        return {
            "feats": torch.ones(1, 8, 263),
            "length": [6],
            "texts": [],
            "joints": None,
        }


def test_run_motiongpt_returns_feats():
    backbone = _StubMotionGPT()
    sample = run_motiongpt(backbone, "jump high", "seq_mg", ref_T=4)
    assert sample.motion is not None
    assert sample.motion.shape == (4, 263)
    assert sample.motion.dtype == np.float32
    assert sample.activations is None
    assert sample.meta["gen_T"] == 6


def test_run_motiongpt_rejects_patched_backbone():
    backbone = _StubMotionGPT()
    backbone._musclemap_bench_eval_patch = True
    try:
        run_motiongpt(backbone, "walk", "s0")
    except RuntimeError as exc:
        assert "patched" in str(exc).lower()
    else:
        raise AssertionError("expected RuntimeError for patched backbone")
