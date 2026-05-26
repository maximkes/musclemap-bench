#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Rajagopal bench names -> Kinesis MyoLeg actuator names (from data/xml/legs/myolegs_assets.xml).
RAJAGOPAL_TO_KINESIS_ACTUATOR: dict[str, str] = {
    "glut_max1_r": "glmax1_r",
    "glut_max2_r": "glmax2_r",
    "glut_max3_r": "glmax3_r",
    "glut_med1_r": "glmed1_r",
    "glut_med2_r": "glmed2_r",
    "glut_med3_r": "glmed3_r",
    "semimem_r": "semimem_r",
    "semiten_r": "semiten_r",
    "bifemlh_r": "bflh_r",
    "bifemsh_r": "bfsh_r",
    "rect_fem_r": "recfem_r",
    "vas_med_r": "vasmed_r",
    "vas_int_r": "vasint_r",
    "vas_lat_r": "vaslat_r",
    "med_gas_r": "gasmed_r",
    "lat_gas_r": "gaslat_r",
    "soleus_r": "soleus_r",
    "tib_post_r": "tibpost_r",
    "tib_ant_r": "tibant_r",
    "per_brev_r": "perbrev_r",
    "per_long_r": "perlong_r",
    "glut_max1_l": "glmax1_l",
    "glut_max2_l": "glmax2_l",
    "glut_max3_l": "glmax3_l",
    "glut_med1_l": "glmed1_l",
    "glut_med2_l": "glmed2_l",
    "glut_med3_l": "glmed3_l",
    "semimem_l": "semimem_l",
    "semiten_l": "semiten_l",
    "bifemlh_l": "bflh_l",
    "bifemsh_l": "bfsh_l",
    "rect_fem_l": "recfem_l",
    "vas_med_l": "vasmed_l",
    "vas_int_l": "vasint_l",
    "vas_lat_l": "vaslat_l",
    "med_gas_l": "gasmed_l",
    "lat_gas_l": "gaslat_l",
    "soleus_l": "soleus_l",
    "tib_post_l": "tibpost_l",
    "tib_ant_l": "tibant_l",
    "per_brev_l": "perbrev_l",
    "per_long_l": "perlong_l",
}

OUTPUT_MUSCLE_NAMES: list[str] = list(RAJAGOPAL_TO_KINESIS_ACTUATOR.keys())

SMPLX_SLICES: dict[str, slice] = {
    "root_orient": slice(0, 3),
    "pose_body": slice(3, 66),
    "trans": slice(309, 312),
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Kinesis precompute."""
    p = argparse.ArgumentParser(description="Precompute Kinesis activations")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--max-samples", type=int, default=None)
    return p.parse_args()


def _bench_root() -> Path:
    """Return musclemap-bench repository root."""
    return Path(__file__).resolve().parent.parent


def _resolve_path(path: str | Path) -> Path:
    """Resolve a config path relative to the bench repo root."""
    p = Path(path)
    if p.is_absolute():
        return p.resolve()
    return (_bench_root() / p).resolve()


def discover_test_sequences(cfg: dict[str, Any], max_samples: int | None) -> list[dict[str, Any]]:
    """Discover test sequences from nested Project-1 layout (activations.npy per folder)."""
    ds_root = _resolve_path(cfg["test_set"]["dataset_root"])
    if not ds_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {ds_root}")

    entries: list[dict[str, Any]] = []
    for act_path in sorted(ds_root.rglob("activations.npy")):
        seq_dir = act_path.parent
        smplx_path = seq_dir / "smplx_322.npy"
        if not smplx_path.is_file():
            continue
        label_path = seq_dir / "semantic_label.txt"
        text = label_path.read_text(encoding="utf-8").strip() if label_path.is_file() else ""
        entries.append({
            "seq_id": seq_dir.name,
            "text": text,
            "smplx_npy": str(smplx_path.resolve()),
            "activations_npy": str(act_path.resolve()),
        })

    if max_samples is not None and entries:
        rng = np.random.default_rng(int(cfg["test_set"]["seed"]))
        n = min(max_samples, len(entries))
        idx = sorted(rng.choice(len(entries), size=n, replace=False).tolist())
        entries = [entries[i] for i in idx]
    return entries


def _load_smplx_motion(path: Path) -> np.ndarray:
    """Load a [T, 322] SMPL-X motion array."""
    arr = np.load(str(path))
    if arr.ndim != 2 or arr.shape[1] != 322:
        raise ValueError(f"Expected smplx_322 [T, 322], got {arr.shape} at {path}")
    return arr.astype(np.float32, copy=False)


def _smplx_to_pose_aa(motion: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert SMPL-X 322-dim motion to SMPL pose_aa [T, 72] and translation [T, 3]."""
    root = motion[:, SMPLX_SLICES["root_orient"]]
    body = motion[:, SMPLX_SLICES["pose_body"]]
    trans = motion[:, SMPLX_SLICES["trans"]]
    pose_aa = np.concatenate([root, body, np.zeros((motion.shape[0], 6), dtype=np.float32)], axis=1)
    if pose_aa.shape[1] != 72:
        raise ValueError(f"pose_aa must be [T, 72], got {pose_aa.shape}")
    return pose_aa.astype(np.float32, copy=False), trans.astype(np.float32, copy=False)


def _root_quat_wxyz(root_orient: np.ndarray) -> np.ndarray:
    """Convert axis-angle root orientation to MuJoCo wxyz quaternion."""
    from scipy.spatial.transform import Rotation as R

    quat_xyzw = R.from_rotvec(root_orient).as_quat()
    return np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
        dtype=np.float32,
    )


def _kinesis_repo(cfg: dict[str, Any]) -> Path:
    """Return the Kinesis repo root (MuJoCo XML only; no Python import from Kinesis)."""
    repo = _resolve_path(cfg["kinesis"]["repo_path"])
    if not repo.is_dir():
        raise FileNotFoundError(
            f"Kinesis repo not found at {repo}. Clone https://github.com/amathislab/Kinesis "
            f"to config kinesis.repo_path."
        )
    return repo


def _require_mujoco() -> None:
    """Fail fast with setup instructions when MuJoCo is not installed."""
    try:
        import mujoco  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "MuJoCo is required for Kinesis precompute. Create the kinesis env:\n"
            "  conda env create -f environment-kinesis.yml\n"
            "  conda activate musclemap-bench-kinesis\n"
            "Then re-run this script."
        ) from exc


def _get_actuator_names(model: Any) -> list[str]:
    """Actuator names from a MuJoCo model (from Kinesis src/env/myolegs_env.py)."""
    actuators: list[str] = []
    for i in range(model.nu):
        if i == model.nu - 1:
            end_p = None
            for el in (
                "name_numericadr",
                "name_textadr",
                "name_tupleadr",
                "name_keyadr",
                "name_pluginadr",
                "name_sensoradr",
            ):
                v = getattr(model, el)
                if np.any(v):
                    end_p = v[0] if end_p is None else min(end_p, v[0])
            if end_p is None:
                end_p = model.nnames
        else:
            end_p = model.name_actuatoradr[i + 1]
        name = model.names[model.name_actuatoradr[i] : end_p].decode("utf-8").rstrip("\x00")
        actuators.append(name)
    return actuators


def _force_to_activation(forces: np.ndarray, model: Any, data: Any) -> list[float]:
    """Map muscle forces to activations (from Kinesis src/env/myolegs_env.py)."""
    import mujoco

    activations: list[float] = []
    for idx_actuator in range(model.nu):
        length = data.actuator_length[idx_actuator]
        lengthrange = model.actuator_lengthrange[idx_actuator]
        velocity = data.actuator_velocity[idx_actuator]
        acc0 = model.actuator_acc0[idx_actuator]
        prmb = model.actuator_biasprm[idx_actuator, :9]
        prmg = model.actuator_gainprm[idx_actuator, :9]
        bias = mujoco.mju_muscleBias(length, lengthrange, acc0, prmb)
        gain = min(-1, mujoco.mju_muscleGain(length, velocity, lengthrange, acc0, prmg))
        activations.append(float(np.clip((forces[idx_actuator] - bias) / gain, 0, 1)))
    return activations


def _target_length_to_activation(lengths: np.ndarray, data: Any, model: Any) -> np.ndarray:
    """Quasi-static muscle activations from target lengths (Kinesis myolegs_env)."""
    forces: list[float] = []
    for idx_actuator in range(model.nu):
        length = data.actuator_length[idx_actuator]
        velocity = data.actuator_velocity[idx_actuator]
        peak_force = model.actuator_biasprm[idx_actuator, 2]
        kp = 5 * peak_force
        kd = 0.1 * kp
        force = kp * (lengths[idx_actuator] - length) - kd * velocity
        forces.append(float(np.clip(force, -peak_force, 0)))
    activations = _force_to_activation(np.asarray(forces), model, data)
    return np.clip(activations, 0, 1)


def _myolegs_xml_path(kinesis_root: Path) -> Path:
    """Return the MyoLeg MuJoCo model XML used by Kinesis."""
    xml_path = kinesis_root / "data" / "xml" / "legs" / "myolegs.xml"
    if not xml_path.is_file():
        raise FileNotFoundError(f"MyoLeg model XML not found: {xml_path}")
    return xml_path


def _simulate_myolegs_activations(
    motion: np.ndarray,
    cfg: dict[str, Any],
    *,
    max_frames: int,
) -> tuple[np.ndarray, list[str]]:
    """Retarget SMPL-X root pose and extract MyoLeg actuator activations via MuJoCo."""
    import mujoco

    kinesis_root = _kinesis_repo(cfg)
    xml_path = _myolegs_xml_path(kinesis_root)

    cwd = os.getcwd()
    try:
        os.chdir(str(xml_path.parent))
        model = mujoco.MjModel.from_xml_path(str(xml_path.name))
    finally:
        os.chdir(cwd)

    data = mujoco.MjData(model)
    actuator_names = _get_actuator_names(model)
    name_to_idx = {n: i for i, n in enumerate(actuator_names)}

    missing = [kin for kin in RAJAGOPAL_TO_KINESIS_ACTUATOR.values() if kin not in name_to_idx]
    if missing:
        raise KeyError(f"Kinesis actuators missing from model: {missing[:5]}")

    pose_aa, trans = _smplx_to_pose_aa(motion)
    t_steps = min(int(pose_aa.shape[0]), int(max_frames))
    if t_steps < 1:
        raise ValueError("Motion has no frames to simulate")

    out = np.zeros((t_steps, len(OUTPUT_MUSCLE_NAMES)), dtype=np.float32)
    neutral = np.asarray(model.key_qpos, dtype=np.float64).reshape(-1).copy()

    for t in range(t_steps):
        qpos = neutral.copy()
        qpos[:3] = trans[t]
        qpos[3:7] = _root_quat_wxyz(pose_aa[t, :3])
        data.qpos[:] = qpos
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        lengths = np.array(data.actuator_length, dtype=np.float64)
        activations = _target_length_to_activation(lengths, data, model)
        for col, raj_name in enumerate(OUTPUT_MUSCLE_NAMES):
            kin_name = RAJAGOPAL_TO_KINESIS_ACTUATOR[raj_name]
            out[t, col] = float(activations[name_to_idx[kin_name]])

    return out, OUTPUT_MUSCLE_NAMES


def run_kinesis_episode(
    seq: dict[str, Any],
    cfg: dict[str, Any],
    muscle_names_cache: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Run Kinesis/MyoLeg forward dynamics and return Rajagopal-aligned activations [T, N]."""
    _ = muscle_names_cache
    motion = _load_smplx_motion(Path(seq["smplx_npy"]))
    max_frames = int(cfg["kinesis"].get("max_frames", 300))
    acts, names = _simulate_myolegs_activations(motion, cfg, max_frames=max_frames)
    return acts.astype(np.float32, copy=False), names


def main() -> None:
    """Precompute Kinesis activation artifacts for the test split."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _require_mujoco()
    args = parse_args()
    cfg = yaml.safe_load(_resolve_path(args.config).read_text(encoding="utf-8"))
    artifact_dir = _resolve_path(cfg["paths"]["kinesis_artifacts"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = artifact_dir / "_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

    seqs = discover_test_sequences(cfg, args.max_samples)
    pending = [s for s in seqs if manifest.get(s["seq_id"], {}).get("status") != "ok"]
    muscle_names: list[str] = []

    for seq in tqdm(pending, desc="Kinesis precompute"):
        seq_id = seq["seq_id"]
        last_error = None
        for attempt in range(int(cfg["kinesis"]["max_retries"])):
            try:
                t0 = __import__("time").perf_counter()
                acts, muscle_names = run_kinesis_episode(seq, cfg, muscle_names)
                elapsed = __import__("time").perf_counter() - t0
                out_name = f"{seq_id}.npy"
                np.save(str(artifact_dir / out_name), acts.astype(np.float32))
                manifest[seq_id] = {"status": "ok", "file": out_name, "timing_s": elapsed}
                break
            except NotImplementedError:
                raise
            except Exception as exc:
                last_error = str(exc)
                logger.warning("Kinesis %s attempt %d failed: %s", seq_id, attempt + 1, last_error)
                if attempt + 1 == int(cfg["kinesis"]["max_retries"]):
                    manifest[seq_id] = {"status": "error", "error": last_error}
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if muscle_names:
        (artifact_dir / "muscle_names.json").write_text(
            json.dumps(muscle_names, indent=2),
            encoding="utf-8",
        )
    logger.info("Done. Manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
