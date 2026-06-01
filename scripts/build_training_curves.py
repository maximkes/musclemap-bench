#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build val training curves CSV from checkpoints.")
    p.add_argument("--config", default="config.yaml")
    p.add_argument(
        "--epochs",
        nargs="*",
        type=int,
        default=None,
        help="Only evaluate these epoch numbers (e.g. 64 69 74). Default: all missing JSON.",
    )
    p.add_argument("--skip-eval", action="store_true", help="Do not call evaluate.py; only aggregate CSV.")
    p.add_argument(
        "--force-eval",
        action="store_true",
        help="Re-run evaluate.py even when val_epoch_* JSON already exists.",
    )
    p.add_argument("--dry-run-eval", action="store_true", help="Pass --dry-run to evaluate.py when invoked.")
    return p.parse_args()


def _metrics_path(results_dir: Path, stem: str, split: str = "val") -> Path:
    return results_dir / f"{split}_{stem}_metrics.json"


def _run_evaluate(
    *,
    evaluate_script: Path,
    train_config: Path,
    ckpt: Path,
    results_dir: Path,
    model_repo: Path,
    dry_run: bool,
) -> bool:
    cmd = [
        sys.executable,
        str(evaluate_script),
        "--config",
        str(train_config),
        "--split",
        "val",
        "--ckpt",
        str(ckpt),
        "--results-dir",
        str(results_dir),
    ]
    if dry_run:
        cmd.append("--dry-run")
    print("[curves] running:", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, check=False, cwd=str(model_repo))
    except OSError as e:
        print(f"[curves] evaluate failed to start: {e}")
        return False
    if proc.returncode != 0:
        print(f"[curves] evaluate exited {proc.returncode} for {ckpt.name}")
        return False
    return True


def main() -> None:
    args = parse_args()
    bench_root = Path(__file__).resolve().parents[1]
    if str(bench_root) not in sys.path:
        sys.path.insert(0, str(bench_root))

    from src.curve_data import (
        build_val_metrics_table,
        discover_checkpoints,
        checkpoint_stem,
    )
    from src.loaders import load_config, _resolve_path

    cfg = load_config(args.config)
    paths = cfg["paths"]
    curve_cfg = cfg.get("training_curve", {})

    ckpt_dir = _resolve_path(paths["checkpoints_dir"])
    curves_dir = _resolve_path(paths["curves_dir"])
    curves_dir.mkdir(parents=True, exist_ok=True)

    model_repo = _resolve_path(paths["musclemap_model_repo"])
    results_dir = model_repo / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    train_config = _resolve_path(paths["musclemap_train_config"])
    train_cfg = yaml.safe_load(train_config.read_text(encoding="utf-8"))
    evaluate_script = model_repo / "scripts" / "evaluate.py"

    checkpoints = discover_checkpoints(ckpt_dir)
    if args.epochs:
        wanted = {int(e) for e in args.epochs}
        checkpoints = [p for p in checkpoints if int(checkpoint_stem(p).split("_")[-1]) in wanted]

    if not checkpoints:
        print(f"[curves] no checkpoints under {ckpt_dir}")
        sys.exit(1)

    print(f"[curves] found {len(checkpoints)} checkpoint(s)")

    if not args.skip_eval:
        for ckpt in checkpoints:
            stem = checkpoint_stem(ckpt)
            out_json = _metrics_path(results_dir, stem)
            if out_json.exists() and not args.force_eval:
                print(f"[curves] skip existing {out_json.name}")
                continue
            if out_json.exists() and args.force_eval:
                print(f"[curves] re-evaluating {out_json.name}")
            if not evaluate_script.exists():
                print(f"[curves] evaluate.py not found at {evaluate_script}")
                break
            ok = _run_evaluate(
                evaluate_script=evaluate_script,
                train_config=train_config,
                ckpt=ckpt,
                results_dir=results_dir,
                model_repo=model_repo,
                dry_run=bool(args.dry_run_eval),
            )
            if not ok:
                print("[curves] stopping eval loop after failure (existing JSON files still aggregated)")

    df = build_val_metrics_table(checkpoints, results_dir, train_cfg, curve_cfg=curve_cfg)
    csv_path = curves_dir / "val_metrics.csv"
    if df.empty:
        print(f"[curves] no val metrics found under {results_dir}")
        sys.exit(1)

    df.to_csv(csv_path, index=False)
    print(f"[curves] wrote {csv_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
