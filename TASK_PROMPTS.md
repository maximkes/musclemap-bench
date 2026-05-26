# Task prompts for Cursor

## Task 1 — inspect dependencies
Read:
- `config.yaml`
- sibling repo in `paths.musclemap_model_repo`
- sibling repo in `paths.musclemap_data_repo`
- Kinesis repo in `kinesis.repo_path`
Then produce a short implementation plan for all remaining `NotImplementedError` blocks.

## Task 2 — implement dataset loading compatibility
Make `src/loaders.py` robust to the actual dataset item structure returned by `MuscleActivationDataset`.
Support either dict-style or tuple-style samples.
Add tests if possible.

## Task 3 — implement MuscleMAP inference compatibility
Inspect the real `MuscleMAPModel.forward()` signature and update `src/inference.py` so `run_musclemap()` works on the actual model.
Do not guess key names.

## Task 4 — implement MotionGPT generation
Inspect the vendored MotionGPT API and implement `run_motiongpt()`.
Return `[T, 263]` motion arrays.

## Task 5 — implement HumanML3D evaluator integration
Inspect the evaluator wrapper in the vendored MotionGPT files and implement:
- `_load_evaluator`
- `extract_motion_features`
- `extract_text_features`
in `src/metrics_l2.py`.

## Task 6 — implement Kinesis precompute
Inspect the actual Kinesis repo and implement `precompute/run_kinesis.py`.
Use explicit retargeting and exact actuator extraction.
Do not assume environment names beyond what is in config.

## Task 7 — improve report quality
Extend `src/report.py` with:
- better captions
- plot titles
- optional CSV summary export
- graceful handling of missing Layer 2

## Task 8 — integration validation
Run:
- `pytest -q`
- smoke benchmark on 3–5 samples
Fix any breakages.
