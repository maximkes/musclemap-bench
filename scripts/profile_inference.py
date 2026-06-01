#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile MuscleMAP inference compute (params / FLOPs).")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--device", default="cpu")
    p.add_argument("--T", type=int, default=196, help="Representative activation sequence length.")
    p.add_argument("--out", default=None, help="JSON output path (default: results/inference_profile.json).")
    return p.parse_args()


def _count_params(model: Any) -> dict[str, int]:
    import torch

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {
        "total_params": int(total),
        "trainable_params": int(trainable),
        "frozen_params": int(frozen),
    }


def _profile_thop(model: Any, device: str, T: int) -> dict[str, float] | None:
    try:
        import torch
        from thop import profile  # type: ignore[import-not-found]
    except ImportError:
        return None

    from src.inference import _ensure_teacher_forced_backbone

    model.to(device)
    model.eval()
    _ensure_teacher_forced_backbone(model)
    text = "a person walks forward slowly"
    with torch.no_grad():
        macs, _params = profile(
            model,
            inputs=([text],),
            kwargs={"motion_tokens": None},
            verbose=False,
        )
    flops = float(macs) * 2.0
    return {
        "macs": float(macs),
        "flops": flops,
        "gflops": flops / 1e9,
        "representative_T": int(T),
        "method": "thop",
    }


def _manual_flops_estimate(model: Any, T: int) -> dict[str, float]:
    """Rough MAC estimate for teacher-forced T5 + activation head (batch=1)."""
    import torch

    hidden = 768
    head_cfg = getattr(model, "config", {}).get("model", {}).get("head", {})
    proj = int(head_cfg.get("proj_dim", 256))
    n_layers = int(head_cfg.get("n_transformer_layers", 3))
    n_heads = int(head_cfg.get("n_heads", 4))
    n_muscles = int(head_cfg.get("n_muscles", 80))
    seq_len = 256

    # T5-base encoder (12 layers) + single decoder step
    enc_layer = 4 * seq_len * hidden * hidden + 2 * seq_len * seq_len * hidden
    dec_layer = 4 * hidden * hidden + 2 * seq_len * hidden
    t5_macs = 12 * enc_layer + dec_layer

    # Activation head transformer over T frames
    head_layer = (
        4 * T * proj * proj
        + 2 * T * T * proj
        + T * proj * n_muscles
    )
    macs = float(t5_macs + n_layers * head_layer)
    flops = macs * 2.0
    return {
        "macs": macs,
        "flops": flops,
        "gflops": flops / 1e9,
        "representative_T": int(T),
        "method": "manual_estimate",
    }


def main() -> None:
    args = parse_args()
    bench_root = Path(__file__).resolve().parents[1]
    if str(bench_root) not in sys.path:
        sys.path.insert(0, str(bench_root))

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    from src.loaders import load_musclemap, _resolve_path

    device = args.device or cfg.get("inference", {}).get("device", "cpu")
    print(f"[profile] loading MuscleMAP on {device}")
    model = load_musclemap(cfg, device=device)

    summary: dict[str, Any] = {
        "musclemap": _count_params(model),
        "kinesis": {
            "total_params": None,
            "note": "Physics simulation (MyoLeg); not a neural forward pass — compare wall-clock only.",
        },
    }

    thop_stats = None
    try:
        thop_stats = _profile_thop(model, device=device, T=args.T)
    except Exception as exc:  # noqa: BLE001
        print(f"[profile] thop profiling failed ({exc}); using manual estimate")
    if thop_stats is not None:
        summary["musclemap"]["compute"] = thop_stats
        print(f"[profile] thop GFLOPs ≈ {thop_stats['gflops']:.3f}")
    else:
        manual = _manual_flops_estimate(model, T=args.T)
        summary["musclemap"]["compute"] = manual
        print(
            "[profile] thop unavailable; wrote manual GFLOPs estimate "
            f"≈ {manual['gflops']:.3f} (pip install thop for measured FLOPs)"
        )

    out_path = Path(args.out) if args.out else _resolve_path(cfg["paths"]["results_dir"]) / "inference_profile.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[profile] wrote {out_path}")


if __name__ == "__main__":
    main()
