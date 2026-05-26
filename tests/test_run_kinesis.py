import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from precompute import run_kinesis as rk


def test_discover_nested_layout(tmp_path: Path) -> None:
    seq_dir = tmp_path / "action_a_clip1"
    seq_dir.mkdir(parents=True)
    np.save(seq_dir / "activations.npy", np.ones((5, 80), dtype=np.float32))
    np.save(seq_dir / "smplx_322.npy", np.ones((5, 322), dtype=np.float32))
    (seq_dir / "semantic_label.txt").write_text("walk forward", encoding="utf-8")

    cfg = {
        "test_set": {"dataset_root": str(tmp_path), "seed": 42},
    }
    found = rk.discover_test_sequences(cfg, max_samples=None)
    assert len(found) == 1
    assert found[0]["seq_id"] == "action_a_clip1"
    assert found[0]["text"] == "walk forward"
    assert Path(found[0]["smplx_npy"]).name == "smplx_322.npy"


def test_smplx_to_pose_aa_shape() -> None:
    motion = np.zeros((10, 322), dtype=np.float32)
    pose_aa, trans = rk._smplx_to_pose_aa(motion)
    assert pose_aa.shape == (10, 72)
    assert trans.shape == (10, 3)


def test_run_kinesis_episode_mocked(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    smplx = tmp_path / "m.npy"
    np.save(smplx, np.zeros((4, 322), dtype=np.float32))
    seq = {"seq_id": "s0", "text": "walk", "smplx_npy": str(smplx)}

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump({
            "kinesis": {"repo_path": str(tmp_path / "Kinesis"), "max_frames": 4, "max_retries": 1},
            "test_set": {"dataset_root": str(tmp_path)},
        }),
        encoding="utf-8",
    )
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    def _fake_sim(motion: np.ndarray, cfg: dict, *, max_frames: int) -> tuple[np.ndarray, list[str]]:
        _ = (cfg, max_frames)
        acts = np.full((motion.shape[0], len(rk.OUTPUT_MUSCLE_NAMES)), 0.5, dtype=np.float32)
        return acts, rk.OUTPUT_MUSCLE_NAMES

    monkeypatch.setattr(rk, "_simulate_myolegs_activations", _fake_sim)
    acts, names = rk.run_kinesis_episode(seq, cfg, [])
    assert acts.shape == (4, len(rk.OUTPUT_MUSCLE_NAMES))
    assert len(rk.OUTPUT_MUSCLE_NAMES) == 42
    assert names == rk.OUTPUT_MUSCLE_NAMES
