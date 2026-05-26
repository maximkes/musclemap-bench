# Implementation plan (Task 1)

Generated after inspecting `config.yaml`, sibling repos, and vendored MotionGPT APIs.
**Do not implement stubs until each task is explicitly started.**

## Dependency verification (2026-05-25)

| Path (from `config.yaml`) | Status |
|---|---|
| `../musclemap-data/output_dataset` | **MISSING** |
| `../musclemap-data/data/motionx_activations` | OK (symlink → `~/Downloads/code/motionx_activations_extended`) |
| `../musclemap-model/config.yaml` | **MISSING** — use `../musclemap-model/config/train.yaml` |
| `../musclemap-model/checkpoints/best.pt` | **MISSING** — use `../checkpoints/epoch_0014.pt` (or latest epoch) |
| `../Kinesis` | **MISSING** — Task 6 blocked until repo is available |
| `vendor/.../finest.tar` (TM2T evaluator) | OK |

### Dataset item structure (Project 1 / `MuscleActivationDataset`)

Per sequence directory (example: `.../Number_12_and_sitting_at_the_same_time_clip1/`):

- `activations.npy` — `[T, 80]` float32 (Rajagopal OpenSim)
- `smplx_322.npy` — `[T, 322]` float32
- `semantic_label.txt` — text label

`MuscleActivationDataset.__getitem__` returns:

```python
{"text", "motion", "acts", "mask", "true_T"}  # padded to max_T=256
```

`run_benchmark.py` currently expects `sequence_id`, `activations` — **adapter needed (Task 2)**.

### Model (Project 2)

- Config: `musclemap-model/config/train.yaml` — `model.head.n_muscles: 80`
- `MuscleMAPModel.forward(text_tokens, motion_tokens=None, ...) -> (logits, pred_log_T, motion_output)`
- Checkpoint: `torch.load(path); state.get("model", state); load_state_dict(..., strict=False)`
- Inference pattern (from `scripts/evaluate.py`): `sigmoid(logits)[0]`, trim with `acts[mask]`

### MotionGPT vendor (Layer 2 / baseline motion)

- `MotionGPT.forward(batch, task="t2m")` → `{"feats": [B,T,263], "length": [B], ...}`
- Generation path: `lm.generate_direct(texts, do_sample=True)` → `vae.decode(tokens)`
- Evaluator: `TM2TMetrics._get_t2m_evaluator` + `get_motion_embeddings` + `t2m_textencoder(word_embs, pos_ohot, text_lengths)`

### Kinesis (Task 6)

Repo not present at `kinesis.repo_path`. `precompute/run_kinesis.py` also assumes flat `activations/*.npy` layout; real data uses nested per-sequence dirs (same as Project 1).

### Already implemented (no stubs)

- `src/align.py`, `src/metrics_l1.py`, `src/report.py`

---

## Prerequisite: fix `config.yaml` before Task 2

```yaml
test_set:
  dataset_root: "../musclemap-data/data/motionx_activations"
paths:
  musclemap_train_config: "../musclemap-model/config/train.yaml"
  musclemap_checkpoint: "../checkpoints/epoch_0014.pt"
```

Keep `kinesis.repo_path` unchanged until the Kinesis repo is cloned.

---

## Stub → API mapping

### 1. Task 2 — `loaders.py` + benchmark sample adapter (no `NotImplementedError`; integration gap)

- `load_test_dataset(cfg)` → `MuscleActivationDataset(Path(dataset_root), config=train_cfg, split=...)`
- Override `train_cfg["data"]["dataset_root"]` from bench config
- Adapter for `run_benchmark.py`:
  - `sequence_id` = `ds._items[idx][0].name`
  - `text` = `sample["text"]`
  - `activations` = `sample["acts"][sample["mask"]].numpy().astype(np.float32)`

### 2. Task 3 — `run_musclemap()` (`src/inference.py`)

| Step | API |
|---|---|
| Forward | `model(text_tokens=[text], motion_tokens=None)` |
| Activations | `torch.sigmoid(logits)[0].cpu().numpy().astype(np.float32)` |
| Length align | `min(pred.shape[0], true_T)` when GT available |
| Optional motion | `motion_output["feats"][0]` if dict (not `"pred_motion"`) |

### 3. Task 4 — `run_motiongpt()` (`src/inference.py`)

| Step | API |
|---|---|
| Load | `load_motiongpt(train_cfg)` from `loaders.py` |
| Generate | `out = backbone.forward({"text": [text], "length": [196]}, task="t2m")` |
| Return | `out["feats"][0, :out["length"][0]]` → `[T, 263]` float32 |

### 4. Task 5 — `src/metrics_l2.py` (3 stubs)

**`_load_evaluator(cfg)`**

- `sys.path.insert(0, musclemap_model/vendor/MotionGPT)`
- Build OmegaConf like `load_motiongpt()` (`METRIC.TM2T` from `config/train.yaml` + `t2m_path` = `deps/t2m/t2m/t2m`)
- `instantiate_from_config(cfg.METRIC.TM2T.t2m_{text,move,motion}encoder)`
- `torch.load(t2m_path/t2m/text_mot_match/model/finest.tar)` → load three encoder state dicts

**`extract_motion_features(motions, cfg)`**

- Normalize each `[T,263]` with `mean.npy` / `std.npy` under `deps/t2m/t2m/t2m/Comp_v6_KLD01/meta/`
- `TM2TMetrics.get_motion_embeddings(feats_batch, lengths)` — `moveencoder(feats[..., :-4])`, `motionencoder(mov, m_lens)`

**`extract_text_features(texts, cfg)`**

- Tokenize like `dataset_t2m_eval.py` / `m2t._get_text_embeddings` (`WordVectorizer`, `word/POS` tokens, max len 20)
- `t2m_textencoder(word_embs, pos_ohot, text_lengths)` → flatten → `[B, D]`

### 5. Task 6 — `run_kinesis_episode()` (`precompute/run_kinesis.py`)

**Blocked** until `../Kinesis` exists.

After clone: inspect retarget + MyoSuite `env_name: myoLegDemo-v0` + actuator logging; fix `discover_test_sequences()` to `rglob("activations.npy")` and per-dir `smplx_322.npy` / `semantic_label.txt`.

### 6. Tasks 7–8

- Task 7: report captions, CSV export, skip Layer 2 gracefully
- Task 8: `pytest -q` + smoke benchmark on 3–5 samples

---

## Execution order

1. Fix `config.yaml` paths (prerequisite)
2. Task 2 → Task 3 → Task 4 → Task 5 (if Layer 2 enabled) → Task 6 (needs Kinesis) → Task 7 → Task 8
3. After **each** coding task: `python -m pytest tests/ -q`
