#!/usr/bin/env python
"""Find the best (checkpoint, val window) pair by per-window MAE and save a visualization."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import yaml

logger = logging.getLogger(__name__)

# Leg-related motions for thesis 3D visualization (sequence_id or prompt).
LEG_MOTION_KEYWORDS: tuple[str, ...] = (
    "walk",
    "walking",
    "jump",
    "jumping",
    "run",
    "running",
    "squat",
    "lunge",
    "kick",
    "hop",
    "rope",
    "stair",
    "leg",
    "sit-to-stand",
    "sit_to_stand",
    "jog",
    "sprint",
    "step",
    "march",
    "skip",
    "skipping",
    "climb",
    "descend",
    "crouch",
    "knee",
    "ankle",
    "stride",
)


def _is_legs_related(sequence_id: str, prompt: str) -> bool:
    hay = f"{sequence_id} {prompt}".lower().replace("_", " ")
    return any(kw in hay for kw in LEG_MOTION_KEYWORDS)


def _mean_joint_displacement(seq_dir: Path, start: int, true_t: int) -> float:
    """Mean per-joint L2 displacement across a val window (SMPL-X FK)."""
    from src.body_visualization import get_smplx_skeleton_joints

    smplx_path = seq_dir / "smplx_322.npy"
    if not smplx_path.is_file():
        return 0.0
    motion = np.load(smplx_path)
    end = min(int(start) + int(true_t), motion.shape[0])
    start_i = int(start)
    if end - start_i < 2:
        return 0.0
    joints = np.stack(
        [get_smplx_skeleton_joints(motion[t]) for t in range(start_i, end)],
        axis=0,
    )
    disp = np.linalg.norm(joints[1:] - joints[:-1], axis=-1)
    return float(disp.mean())


def _window_mae(pred: np.ndarray, true: np.ndarray) -> tuple[float, float]:
    n = min(pred.shape[0], true.shape[0])
    if n == 0:
        return float("nan"), float("nan")
    diff = pred[:n] - true[:n]
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    return mae, rmse


def _is_numeric_blob(text: str) -> bool:
    import re

    tokens = [t for t in re.split(r"[\s,]+", text.strip()) if t]
    if not tokens:
        return True
    numeric = sum(1 for t in tokens if _token_is_float(t))
    return (numeric / len(tokens)) > 0.5


def _token_is_float(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def _read_prompt(seq_dir: Path, fallback_text: str) -> str:
    label_path = seq_dir / "semantic_label.txt"
    if label_path.is_file():
        text = label_path.read_text(encoding="utf-8").strip()
        if text and not _is_numeric_blob(text):
            return text
    return fallback_text


def _plot_example(
    *,
    pred: np.ndarray,
    true: np.ndarray,
    muscle_names: list[str],
    title: str,
    out_path: Path,
    top_k: int = 20,
) -> None:
    n = min(pred.shape[0], true.shape[0])
    pred = pred[:n]
    true = true[:n]
    mean_act = np.maximum(pred.mean(axis=0), true.mean(axis=0))
    idx = np.argsort(mean_act)[::-1][:top_k]
    labels = [muscle_names[int(i)] for i in idx]

    fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    for ax, data, ylab in (
        (axes[0], true[:, idx].T, "Ground truth"),
        (axes[1], pred[:, idx].T, "MuscleMAP prediction"),
    ):
        im = ax.imshow(data, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_ylabel(ylab)
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    axes[1].set_xlabel("Frame (window)")
    fig.suptitle(title, fontsize=10, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pick best val example across checkpoints.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--split", default="val", choices=("val", "test"))
    parser.add_argument(
        "--checkpoints-dir",
        default=None,
        help="Override config paths.checkpoints_dir",
    )
    parser.add_argument("--device", default=None, help="cpu | cuda | mps (default: auto)")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Only scan this checkpoint stem (e.g. epoch_0019 or epoch_0019.pt)",
    )
    parser.add_argument(
        "--min-joint-displacement",
        type=float,
        default=0.002,
        help="Skip windows with mean joint displacement below this (meters, FK)",
    )
    parser.add_argument(
        "--legs-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only consider leg-related motions (walk, jump, run, etc.)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Default: results/best_val_example.json",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=None,
        help="Default: results/plots/best_val_example.png",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    bench_root = Path(__file__).resolve().parents[1]
    if str(bench_root) not in sys.path:
        sys.path.insert(0, str(bench_root))

    from src.curve_data import discover_checkpoints, checkpoint_stem
    from src.inference import run_musclemap
    from src.loaders import load_config, _resolve_path, load_musclemap, _import_from_model_repo

    cfg = load_config(args.config)
    if args.checkpoints_dir:
        ckpt_dir = Path(args.checkpoints_dir).resolve()
    else:
        ckpt_dir = _resolve_path(cfg["paths"]["checkpoints_dir"])

    results_dir = _resolve_path(cfg["paths"]["results_dir"])
    plots_dir = results_dir / "plots"
    out_json = args.output_json or (results_dir / "best_val_example.json")
    out_png = args.output_png or (plots_dir / "best_val_example.png")

    train_config = _resolve_path(cfg["paths"]["musclemap_train_config"])
    train_cfg = yaml.safe_load(train_config.read_text(encoding="utf-8"))
    dataset_root = Path(str(train_cfg["data"]["dataset_root"]))

    dataset_mod = _import_from_model_repo(cfg["paths"]["musclemap_model_repo"], "src.dataset")
    MuscleActivationDataset = dataset_mod.MuscleActivationDataset
    ds = MuscleActivationDataset(dataset_root, config=train_cfg, split=args.split)
    muscle_names = list(ds.muscle_names)

    import torch

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    checkpoints = discover_checkpoints(ckpt_dir)
    if args.checkpoint:
        stem = Path(str(args.checkpoint)).stem
        checkpoints = [p for p in checkpoints if p.stem == stem]
        if not checkpoints:
            raise SystemExit(f"No checkpoint matching {args.checkpoint!r} under {ckpt_dir}")
    if not checkpoints:
        raise SystemExit(f"No checkpoints under {ckpt_dir}")

    best: dict[str, Any] | None = None
    model = None

    for ckpt_path in checkpoints:
        stem = checkpoint_stem(ckpt_path)
        epoch = int(stem.split("_")[-1])
        logger.info("Scanning %s (%d val windows)", stem, len(ds))
        if model is None:
            model = load_musclemap(cfg, device=device, checkpoint_path=ckpt_path)
        else:
            state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            sd = state.get("model", state)
            model.load_state_dict(sd, strict=False)
            model.to(device)
            model.eval()

        for i in range(len(ds)):
            seq_dir, start, true_t, text = ds._items[i]
            sample = ds[i]
            mask = sample["mask"]
            true = sample["acts"][mask].cpu().numpy().astype(np.float32, copy=False)
            seq_id = seq_dir.name
            prompt = _read_prompt(seq_dir, str(sample["text"]))
            if "standing" in seq_id.lower() or "static" in seq_id.lower():
                continue
            if args.legs_only and not _is_legs_related(seq_id, prompt):
                continue
            joint_disp = _mean_joint_displacement(seq_dir, int(start), int(true_t))
            if joint_disp < float(args.min_joint_displacement):
                continue

            mm = run_musclemap(model, str(sample["text"]), seq_id, device=device, ref_T=int(true_t))
            pred = mm.activations
            if pred is None:
                continue
            mae, rmse = _window_mae(pred, true)
            if best is None or mae < float(best["mae"]):
                best = {
                    "checkpoint": stem,
                    "checkpoint_path": str(ckpt_path.resolve()),
                    "epoch": epoch,
                    "split": args.split,
                    "dataset_index": i,
                    "sequence_id": seq_id,
                    "sequence_dir": str(seq_dir.resolve()),
                    "window_start": int(start),
                    "true_T": int(true_t),
                    "prompt": prompt,
                    "dataset_text": str(sample["text"]),
                    "mae": mae,
                    "rmse": rmse,
                    "joint_displacement": joint_disp,
                }
                logger.info(
                    "  new best: mae=%.6f ckpt=%s seq=%s prompt=%r",
                    mae,
                    stem,
                    seq_id,
                    prompt[:60],
                )

        _maybe_empty_cache()

    if best is None:
        raise SystemExit("No valid windows scored")

    logger.info(
        "Global best: %s | %s | MAE=%.6f",
        best["checkpoint"],
        best["sequence_id"],
        best["mae"],
    )

    # Re-run once to save arrays and PNG for the winner (reuse loaded weights).
    ckpt_path = Path(best["checkpoint_path"])
    if model is None:
        model = load_musclemap(cfg, device=device, checkpoint_path=ckpt_path)
    elif checkpoint_stem(ckpt_path) != best["checkpoint"]:
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        model.load_state_dict(state.get("model", state), strict=False)
        model.to(device)
        model.eval()
    sample = ds[int(best["dataset_index"])]
    mask = sample["mask"]
    true = sample["acts"][mask].cpu().numpy().astype(np.float32, copy=False)
    mm = run_musclemap(
        model,
        str(sample["text"]),
        str(best["sequence_id"]),
        device=device,
        ref_T=int(best["true_T"]),
    )
    pred = mm.activations
    assert pred is not None

    npy_dir = results_dir / "best_val_example"
    npy_dir.mkdir(parents=True, exist_ok=True)
    pred_path = npy_dir / "pred_activations.npy"
    true_path = npy_dir / "true_activations.npy"
    np.save(pred_path, pred)
    np.save(true_path, true)
    best["pred_activations_npy"] = str(pred_path.resolve())
    best["true_activations_npy"] = str(true_path.resolve())

    title = (
        f"{best['prompt'][:100]}\n"
        f"checkpoint={best['checkpoint']}  seq={best['sequence_id']}  "
        f"MAE={best['mae']:.4f}  RMSE={best['rmse']:.4f}"
    )
    _plot_example(
        pred=pred,
        true=true,
        muscle_names=muscle_names,
        title=title,
        out_path=out_png,
    )
    best["visualization_png"] = str(out_png.resolve())

    out_json.write_text(json.dumps(best, indent=2), encoding="utf-8")

    try:
        from src.body_visualization import load_best_example_arrays, render_activation_skeleton_montage

        smplx_win, pred_body, names_body, _seq_dir = load_best_example_arrays(best, cfg, train_cfg)
        body_png = plots_dir / "best_val_example_body.png"
        render_activation_skeleton_montage(
            smplx_win,
            pred_body,
            names_body,
            body_png,
            title=(
                f"{best['prompt']}\n"
                f"{best['checkpoint']} · MAE={float(best['mae']):.4f}"
            ),
        )
        best["body_visualization_png"] = str(body_png.resolve())
        out_json.write_text(json.dumps(best, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("Body montage skipped: %s", exc)
    print(json.dumps(best, indent=2))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_png}")


def _maybe_empty_cache() -> None:
    import gc

    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
