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
├── scripts/
│   ├── build_muscle_mapping.py
│   ├── quick_smoke.py
│   ├── render_report.py
│   └── run_benchmark.py
├── src/
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
