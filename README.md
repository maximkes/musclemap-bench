# musclemap-bench

Benchmark suite for the thesis pipeline:
- **Project 1:** `musclemap-data`
- **Project 2:** `musclemap-model`
- **Project 3:** `musclemap-bench`

`musclemap-bench` compares:
1. **Biomechanical accuracy** — MuscleMAP vs Kinesis against OpenSim ground truth.
2. **Text-to-motion quality** — MuscleMAP vs MotionGPT in the HumanML3D evaluator space.
3. **Compute cost** — training and inference resource summary.

## Naming scheme
The repo is intentionally named **musclemap-bench** to match the previous two parts:
- `musclemap-data`
- `musclemap-model`
- `musclemap-bench`

## Status
This repo is a strong, production-minded scaffold. Some integrations are intentionally left behind explicit `NotImplementedError` markers because they depend on your exact local copies of:
- Kinesis
- MyoSuite environment names
- MotionGPT vendor layout
- HumanML3D evaluator checkpoint structure

Those places are documented with exact hints so Cursor can implement them safely.

## Environments
### Main benchmark env
```bash
conda env create -f environment.yml
conda activate musclemap-bench
```

### Separate Kinesis env
```bash
conda env create -f environment-kinesis.yml
conda activate musclemap-bench-kinesis
```

## Quick start
### 1. Build the muscle mapping file
```bash
python scripts/build_muscle_mapping.py --config config.yaml
```

### 2. Optional: precompute Kinesis artifacts
```bash
conda activate musclemap-bench-kinesis
python precompute/run_kinesis.py --config config.yaml --max-samples 10
```

### 3. Run a smoke benchmark
```bash
conda activate musclemap-bench
python scripts/quick_smoke.py --config config.yaml
```

### 4. Run the full benchmark
```bash
python scripts/run_benchmark.py --config config.yaml --device cpu
```

### 5. Thesis figures (`notebooks/thesis_figures.ipynb`)
Refresh validation training curves, inference timings, and thesis plots:

```bash
# Aggregate val MAE/RMSE per checkpoint (skips existing val_epoch_* JSON)
python scripts/build_training_curves.py --config config.yaml

# Re-evaluate all checkpoints and refresh val_metrics.csv (e.g. after adding epoch_0089+)
python scripts/build_training_curves.py --config config.yaml --force-eval

# Best val window across all checkpoints (prompt + heatmap for thesis)
conda run -n musclemap-model python scripts/pick_best_val_example.py --config config.yaml

# Optional: evaluate missing checkpoints only (e.g. epochs 64 69 74)
python scripts/build_training_curves.py --config config.yaml --epochs 64 69 74

# Per-request inference timings for Figure B
python scripts/run_benchmark.py --config config.yaml --device cpu --export-timings

# Parameter / FLOPs profile for Figure B
python scripts/profile_inference.py --config config.yaml --device cpu
```

Then open `notebooks/thesis_figures.ipynb` (run with cwd = `musclemap-bench`). Figures are written to `results/plots/`.

### 6. Kinesis triple comparison (`notebooks/kinesis_comparison.ipynb`)

Lists test movements with Kinesis artifacts, renders side-by-side skeleton montages (OpenSim GT · MuscleMAP · Kinesis), and exports per-window metrics to `results/kinesis_triple_comparison.csv`.

```bash
conda activate musclemap-bench
python scripts/rebuild_kinesis_comparison_nb.py   # regenerate notebook source
jupyter notebook notebooks/kinesis_comparison.ipynb
```

Set `MAX_EVAL` / `MAX_VIS` at the top of the notebook for a quick smoke run.

Set `resources.musclemap_training_hours` in `config.yaml` from your DataSphere wall-clock time before exporting Figure C (training GPU-hours vs paired test MAE). Kinesis training uses `resources.kinesis_training_hours` (default 240 h on 1× A100, paper cite).

**Note:** Validation curves use val-split MAE/RMSE; Kinesis horizontal baselines in Figure A are **test** paired metrics from `baselines.kinesis_test_paired` (intentional target line — document in caption).

## Repository layout
```text
musclemap-bench/
├── config.yaml
├── environment.yml
├── environment-kinesis.yml
├── pyproject.toml
├── README.md
├── precompute/
│   └── run_kinesis.py
├── data/
│   └── curves/              # val_metrics.csv from build_training_curves.py
├── notebooks/
│   ├── plot_results.ipynb
│   ├── thesis_figures.ipynb
│   └── kinesis_comparison.ipynb   # GT vs MuscleMAP vs Kinesis (list, viz, metrics)
├── scripts/
│   ├── build_muscle_mapping.py
│   ├── build_training_curves.py
│   ├── profile_inference.py
│   ├── quick_smoke.py
│   ├── render_report.py
│   └── run_benchmark.py
├── src/
│   ├── curve_data.py
│   ├── __init__.py
│   ├── align.py
│   ├── inference.py
│   ├── loaders.py
│   ├── metrics_l1.py
│   ├── metrics_l2.py
│   ├── report.py
│   └── resources.py
├── tests/
│   ├── test_align.py
│   └── test_metrics_l1.py
└── .cursor/
    └── rules/
```

## Cursor workflow
1. Open this repo in Cursor.
2. First give Cursor `CURSOR_MAIN_PROMPT.md`.
3. Then give it tasks from `TASK_PROMPTS.md` in order.
4. After each task, run:
```bash
pytest -q
```

## Implementation policy
- Never silently guess Kinesis retargeting internals.
- Never silently guess MotionGPT evaluator API.
- Where an integration is uncertain, fail loudly with a specific message.
- Keep all benchmark metrics deterministic.
- Keep paths OS-safe via `pathlib.Path`.
