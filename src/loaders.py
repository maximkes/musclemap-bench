from __future__ import annotations

import importlib
import json
import logging
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BenchmarkSample:
    """Normalized benchmark row from a dataset index."""

    sequence_id: str
    text: str
    activations: np.ndarray
    motion: np.ndarray | None = None


def _bench_root() -> Path:
    """Return musclemap-bench repository root."""
    return Path(__file__).resolve().parent.parent


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a dict."""
    with Path(path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a mapping")
    return cfg


def _resolve_path(path: str | Path) -> Path:
    """Resolve a config path relative to the bench repo root."""
    p = Path(path)
    if p.is_absolute():
        return p.resolve()
    return (_bench_root() / p).resolve()


def _to_numpy_f32(value: Any) -> np.ndarray:
    """Convert a tensor or array to float32 numpy."""
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(value, dtype=np.float32)


def _masked_activations(acts: Any, mask: Any) -> np.ndarray:
    """Slice padded activations to valid frames using a boolean mask."""
    acts_np = _to_numpy_f32(acts)
    if hasattr(mask, "detach"):
        mask_np = mask.detach().cpu().numpy().astype(bool, copy=False)
    else:
        mask_np = np.asarray(mask, dtype=bool)
    if acts_np.ndim != 2:
        raise ValueError(f"activations must be 2D, got shape {acts_np.shape}")
    if mask_np.shape[0] != acts_np.shape[0]:
        raise ValueError(f"mask length {mask_np.shape[0]} != acts frames {acts_np.shape[0]}")
    return acts_np[mask_np].astype(np.float32, copy=False)


def _optional_motion(raw: dict[str, Any], mask: Any | None) -> np.ndarray | None:
    """Extract optional [T, 322] motion features when present."""
    if "motion" not in raw:
        return None
    motion = _to_numpy_f32(raw["motion"])
    if motion.ndim != 2 or motion.shape[1] != 322:
        raise ValueError(f"motion must be [T, 322], got {motion.shape}")
    if mask is None:
        return motion
    if hasattr(mask, "detach"):
        mask_np = mask.detach().cpu().numpy().astype(bool, copy=False)
    else:
        mask_np = np.asarray(mask, dtype=bool)
    return motion[mask_np].astype(np.float32, copy=False)


def normalize_dataset_sample(
    raw: dict[str, Any] | tuple[Any, ...],
    *,
    sequence_id: str = "",
) -> BenchmarkSample:
    """Normalize dict- or tuple-style dataset items into a benchmark sample."""
    if isinstance(raw, tuple):
        return _normalize_tuple_sample(raw, sequence_id=sequence_id)
    if not isinstance(raw, dict):
        raise TypeError(f"Unsupported dataset sample type: {type(raw)!r}")

    text = str(raw.get("text", "")).strip()
    mask = raw.get("mask")

    if "activations" in raw:
        activations = _to_numpy_f32(raw["activations"])
    elif "acts" in raw and mask is not None:
        activations = _masked_activations(raw["acts"], mask)
    elif "acts" in raw:
        activations = _to_numpy_f32(raw["acts"])
        true_t = raw.get("true_T")
        if true_t is not None:
            activations = activations[: int(true_t)]
    else:
        raise KeyError("Sample must provide 'activations' or 'acts' (+ optional 'mask')")

    if activations.ndim != 2:
        raise ValueError(f"activations must be 2D, got shape {activations.shape}")

    motion = _optional_motion(raw, mask)
    return BenchmarkSample(
        sequence_id=sequence_id,
        text=text,
        activations=activations.astype(np.float32, copy=False),
        motion=motion,
    )


def _normalize_tuple_sample(raw: tuple[Any, ...], *, sequence_id: str) -> BenchmarkSample:
    """Normalize tuple samples using MuscleActivationDataset field order."""
    n = len(raw)
    if n == 5:
        text, motion, acts, mask, _true_t = raw
        sample = normalize_dataset_sample(
            {"text": text, "motion": motion, "acts": acts, "mask": mask},
            sequence_id=sequence_id,
        )
        return sample
    if n == 4:
        text, acts, mask, _true_t = raw
        return normalize_dataset_sample(
            {"text": text, "acts": acts, "mask": mask},
            sequence_id=sequence_id,
        )
    if n == 3:
        text, acts, mask = raw
        return normalize_dataset_sample(
            {"text": text, "acts": acts, "mask": mask},
            sequence_id=sequence_id,
        )
    if n == 2:
        a, b = raw
        if isinstance(a, str):
            return normalize_dataset_sample({"text": a, "acts": b}, sequence_id=sequence_id)
        if isinstance(b, str):
            return normalize_dataset_sample({"text": b, "acts": a}, sequence_id=sequence_id)
        raise ValueError("2-tuple sample must contain one string label and activations")
    raise ValueError(f"Unsupported tuple sample length: {n}")


def _sequence_id_for_index(dataset: Any, index: int) -> str:
    """Read the sequence directory name for a dataset index when available."""
    items = getattr(dataset, "_items", None)
    if items is not None and 0 <= index < len(items):
        seq_dir = items[index][0]
        return Path(seq_dir).name
    return str(index)


def get_benchmark_sample(dataset: Any, index: int) -> BenchmarkSample:
    """Fetch and normalize one dataset row for benchmarking."""
    sequence_id = _sequence_id_for_index(dataset, index)
    raw = dataset[index]
    return normalize_dataset_sample(raw, sequence_id=sequence_id)


def _inject_repo(repo_path: str | Path) -> None:
    """Prepend a sibling repo to sys.path for imports."""
    p = str(_resolve_path(repo_path))
    if p not in sys.path:
        sys.path.insert(0, p)


def _import_from_model_repo(repo_path: str | Path, import_stmt: str) -> Any:
    """Import from musclemap-model without shadowing by this bench's ``src`` package."""
    repo = str(_resolve_path(repo_path))
    bench_root = str(_bench_root())
    saved_modules = {
        key: sys.modules[key]
        for key in list(sys.modules)
        if key == "src" or key.startswith("src.")
    }
    saved_path = list(sys.path)
    try:
        for key in saved_modules:
            del sys.modules[key]
        sys.path = [p for p in sys.path if p not in (repo, bench_root)]
        sys.path.insert(0, repo)
        return importlib.import_module(import_stmt)
    finally:
        for key in list(sys.modules):
            if key == "src" or key.startswith("src."):
                del sys.modules[key]
        sys.modules.update(saved_modules)
        sys.path[:] = saved_path


def _resolve_model_train_cfg(cfg: dict[str, Any], model_train_cfg: dict[str, Any]) -> dict[str, Any]:
    """Resolve musclemap-model-relative paths in the training config."""
    repo = _resolve_path(cfg["paths"]["musclemap_model_repo"])
    resolved = dict(model_train_cfg)
    model_block = dict(resolved.get("model", {}))
    mg_dir = model_block.get("motiongpt_dir")
    if isinstance(mg_dir, str) and mg_dir and not Path(mg_dir).is_absolute():
        model_block["motiongpt_dir"] = str((repo / mg_dir).resolve())
    resolved["model"] = model_block
    return resolved


def load_test_dataset(cfg: dict[str, Any]) -> tuple[Any, list[str]]:
    """Load the test split MuscleActivationDataset and Rajagopal muscle names."""
    dataset_root = _resolve_path(cfg["test_set"]["dataset_root"])
    train_config_path = _resolve_path(cfg["paths"]["musclemap_train_config"])
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not train_config_path.exists():
        raise FileNotFoundError(f"Training config not found: {train_config_path}")

    model_train_cfg = load_config(train_config_path)
    model_train_cfg["data"]["dataset_root"] = str(dataset_root)
    dataset_mod = _import_from_model_repo(cfg["paths"]["musclemap_model_repo"], "src.dataset")
    MuscleActivationDataset = dataset_mod.MuscleActivationDataset

    ds = MuscleActivationDataset(dataset_root, config=model_train_cfg, split=cfg["test_set"]["split"])
    logger.info("Loaded test dataset: %d items from %s", len(ds), dataset_root)
    return ds, list(ds.muscle_names)


def load_motiongpt_backbone(cfg: dict[str, Any], device: str = "cpu") -> Any:
    """Load the frozen MotionGPT backbone for baseline motion generation."""
    if "bert_score" not in sys.modules:
        import torch
        mod = types.ModuleType("bert_score")
        mod.score = lambda *a, **k: (torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]))
        sys.modules["bert_score"] = mod
    model_mod = _import_from_model_repo(cfg["paths"]["musclemap_model_repo"], "src.model")
    load_motiongpt = model_mod.load_motiongpt

    model_train_cfg = _resolve_model_train_cfg(
        cfg, load_config(_resolve_path(cfg["paths"]["musclemap_train_config"]))
    )
    backbone = load_motiongpt(model_train_cfg)
    backbone.to(device)
    backbone.eval()
    return backbone


def load_musclemap(
    cfg: dict[str, Any],
    device: str = "cpu",
    *,
    checkpoint_path: str | Path | None = None,
):
    """Load the MuscleMAP model and checkpoint."""
    ckpt = _resolve_path(checkpoint_path) if checkpoint_path is not None else _resolve_path(
        cfg["paths"]["musclemap_checkpoint"]
    )
    if not ckpt.exists():
        raise FileNotFoundError(f"MuscleMAP checkpoint not found: {ckpt}")

    if "bert_score" not in sys.modules:
        import torch
        mod = types.ModuleType("bert_score")
        mod.score = lambda *a, **k: (torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]))
        sys.modules["bert_score"] = mod
    import torch
    model_mod = _import_from_model_repo(cfg["paths"]["musclemap_model_repo"], "src.model")
    MuscleMAPModel = model_mod.MuscleMAPModel
    load_motiongpt = model_mod.load_motiongpt

    model_train_cfg = _resolve_model_train_cfg(
        cfg, load_config(_resolve_path(cfg["paths"]["musclemap_train_config"]))
    )
    backbone = load_motiongpt(model_train_cfg)
    model = MuscleMAPModel(backbone=backbone, config=model_train_cfg)
    state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    sd = state.get("model", state)
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return model


def load_kinesis_manifest(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Load the Kinesis precompute manifest keyed by sequence id."""
    artifact_dir = _resolve_path(cfg["paths"]["kinesis_artifacts"])
    manifest_path = artifact_dir / "_manifest.json"
    if not manifest_path.exists():
        return {}
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for seq_id, info in raw.items():
        if info.get("status") == "ok" and "file" in info:
            item = dict(info)
            item["path"] = str((artifact_dir / info["file"]).resolve())
            out[seq_id] = item
    return out


def load_kinesis_activations(npy_path: str | Path) -> np.ndarray:
    """Load Kinesis activation array from a precomputed npy file."""
    return np.load(str(npy_path)).astype(np.float32)


def load_kinesis_muscle_names(cfg: dict[str, Any]) -> list[str]:
    """Load Kinesis muscle name list from precompute artifacts."""
    p = _resolve_path(cfg["paths"]["kinesis_artifacts"]) / "muscle_names.json"
    if not p.exists():
        raise FileNotFoundError(f"Kinesis muscle_names.json not found at {p}")
    return json.loads(p.read_text(encoding="utf-8"))
