from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def clean_label(sample_id: str) -> str:
    """Convert a sequence id into a canonical action label (matches musclemap-model)."""
    s = re.sub(r"_clip\d+$", "", sample_id, flags=re.IGNORECASE)
    s = re.sub(r"_\d+$", "", s)
    return s.replace("_", " ").strip().lower()


def _is_numeric_blob(text: str) -> bool:
    tokens = [t for t in re.split(r"[\s,]+", text.strip()) if t]
    if not tokens:
        return True
    numeric = sum(1 for t in tokens if _is_float_token(t))
    return (numeric / len(tokens)) > 0.5


def _is_float_token(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def _scan_sequence_dirs(dataset_root: Path, *, min_T: int) -> list[tuple[Path, int, str]]:
    seq_infos: list[tuple[Path, int, str]] = []
    for act_path in sorted(dataset_root.rglob("activations.npy")):
        seq_dir = act_path.parent
        if not (seq_dir / "smplx_322.npy").is_file():
            continue
        try:
            acts_np = np.load(act_path)
        except (EOFError, ValueError, OSError, TimeoutError):
            continue
        if acts_np.ndim != 2:
            continue
        t = int(acts_np.shape[0])
        if t < min_T:
            continue
        label_path = seq_dir / "semantic_label.txt"
        label_text = label_path.read_text(encoding="utf-8").strip() if label_path.is_file() else ""
        if not label_text or _is_numeric_blob(label_text):
            text = clean_label(seq_dir.name)
        else:
            text = label_text.strip().lower()
        canonical = clean_label(text.replace(" ", "_"))
        seq_infos.append((seq_dir, t, canonical))
    return seq_infos


def test_split_groups(data_cfg: dict[str, Any]) -> set[str]:
    """Return canonical group names assigned to the test split."""
    groups: dict[str, list[tuple[Path, int]]] = {}
    for seq_dir, _t, canonical in _scan_sequence_dirs(
        Path(data_cfg["dataset_root"]),
        min_T=int(data_cfg.get("min_T", 30)),
    ):
        groups.setdefault(canonical, []).append((seq_dir, _t))

    split_seed = int(data_cfg.get("split_seed", 42))
    rng = random.Random(split_seed)
    canonical_names = sorted(groups.keys())
    rng.shuffle(canonical_names)

    train_p = float(data_cfg.get("train_split", 0.90))
    val_p = float(data_cfg.get("val_split", 0.05))
    n_groups = len(canonical_names)
    n_train = min(int(round(train_p * n_groups)), n_groups)
    n_val = min(int(round(val_p * n_groups)), n_groups - n_train)
    return set(canonical_names[n_train + n_val :])


def discover_test_split_sequence_entries(
    cfg: dict[str, Any],
    *,
    bench_root: Path,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    """List unique test-split sequences without importing torch / MuscleActivationDataset."""
    train_cfg_path = Path(cfg["paths"]["musclemap_train_config"])
    if not train_cfg_path.is_absolute():
        train_cfg_path = (bench_root / train_cfg_path).resolve()
    train_cfg = yaml.safe_load(train_cfg_path.read_text(encoding="utf-8"))
    data_cfg = dict(train_cfg["data"])

    ds_root = Path(cfg["test_set"]["dataset_root"])
    if not ds_root.is_absolute():
        ds_root = (bench_root / ds_root).resolve()
    data_cfg["dataset_root"] = str(ds_root)

    test_groups = test_split_groups(data_cfg)
    min_t = int(data_cfg.get("min_T", 30))
    seen: set[str] = set()
    entries: list[dict[str, Any]] = []
    for seq_dir, _t, canonical in _scan_sequence_dirs(ds_root, min_T=min_t):
        if canonical not in test_groups:
            continue
        seq_id = seq_dir.name
        if seq_id in seen:
            continue
        seen.add(seq_id)
        label_path = seq_dir / "semantic_label.txt"
        text = label_path.read_text(encoding="utf-8").strip() if label_path.is_file() else clean_label(seq_id)
        entries.append(
            {
                "seq_id": seq_id,
                "text": text,
                "smplx_npy": str((seq_dir / "smplx_322.npy").resolve()),
                "activations_npy": str((seq_dir / "activations.npy").resolve()),
            }
        )

    entries.sort(key=lambda e: e["seq_id"])
    if max_samples is not None and entries:
        rng = np.random.default_rng(int(cfg["test_set"]["seed"]))
        n = min(max_samples, len(entries))
        idx = sorted(rng.choice(len(entries), size=n, replace=False).tolist())
        entries = [entries[i] for i in idx]
    return entries
