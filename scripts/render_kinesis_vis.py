#!/usr/bin/env python3
"""Render triple GT / MuscleMAP / Kinesis animation for one sequence id."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from precompute.run_kinesis import RAJAGOPAL_TO_KINESIS_ACTUATOR
from src.align import build_mapping_kinesis_artifacts
from src.body_visualization import create_triple_activation_animation, save_animation_media
from src.loaders import load_config, load_kinesis_manifest, load_kinesis_muscle_names, load_musclemap, load_test_dataset
from src.paired_eval import build_triple_vis_example


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sequence-id", default="Simultaneously_Waddling_and_walking_clip1")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--frame-step", type=int, default=1)
    args = p.parse_args()

    cfg = load_config(_REPO / args.config)
    device = str(cfg.get("inference", {}).get("device", "cpu"))
    seq_id = args.sequence_id

    print(f"Loading dataset and model (device={device})...")
    dataset, rajagopal_names = load_test_dataset(cfg)
    k_names = load_kinesis_muscle_names(cfg)
    mapping = build_mapping_kinesis_artifacts(rajagopal_names, k_names, RAJAGOPAL_TO_KINESIS_ACTUATOR)
    k_manifest = load_kinesis_manifest(cfg)
    model = load_musclemap(cfg, device=device)

    print(f"Building vis arrays for {seq_id!r}...")
    ex = build_triple_vis_example(
        seq_id,
        cfg=cfg,
        dataset=dataset,
        model=model,
        k_manifest=k_manifest,
        mapping=mapping,
        rajagopal_names=rajagopal_names,
        device=device,
    )
    print(f"Frames: {ex['smplx'].shape[0]}, prompt: {ex['prompt'][:80]!r}")

    out_dir = _REPO / "results" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"triple_{seq_id}.mp4"

    print("Rendering animation...")
    anim = create_triple_activation_animation(
        ex["smplx"],
        ex["gt"],
        ex["mm"],
        ex["kin_full"],
        rajagopal_names,
        fps=args.fps,
        frame_step=args.frame_step,
        title=f"{seq_id}\n{ex['prompt'][:80]}",
    )
    saved = save_animation_media(anim, out_path, fps=args.fps)
    print(f"Saved {saved} ({saved.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
