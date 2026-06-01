"""3D SMPL-X skeleton on a human figure, colored by predicted muscle activations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.spatial.transform import Rotation as R

# Motion-X++ layout (must match musclemap-data/src/smplx_to_opensim.py).
_SMPLX_SLICES = {
    "root_orient": slice(0, 3),
    "pose_body": slice(3, 66),
    "trans": slice(309, 312),
}

_SMPL_PARENTS = np.array(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
    dtype=int,
)

_SMPL_OFFSETS = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.09, -0.09, 0.0],
        [-0.09, -0.09, 0.0],
        [0.0, 0.12, 0.0],
        [0.0, -0.4, 0.0],
        [0.0, -0.4, 0.0],
        [0.0, 0.12, 0.0],
        [0.0, -0.4, 0.0],
        [0.0, -0.4, 0.0],
        [0.0, 0.15, 0.0],
        [0.0, -0.06, 0.1],
        [0.0, -0.06, 0.1],
        [0.0, 0.18, 0.0],
        [-0.06, 0.05, 0.0],
        [0.06, 0.05, 0.0],
        [0.0, 0.12, 0.0],
        [-0.15, 0.0, 0.0],
        [0.15, 0.0, 0.0],
        [0.0, -0.27, 0.0],
        [0.0, -0.27, 0.0],
        [0.0, -0.25, 0.0],
        [0.0, -0.25, 0.0],
        [0.0, -0.1, 0.0],
        [0.0, -0.1, 0.0],
    ],
    dtype=np.float64,
)

def _muscle_side(name: str) -> str | None:
    if name.endswith("_r"):
        return "r"
    if name.endswith("_l"):
        return "l"
    return None


def _joint_pair_for_muscle(name: str) -> tuple[int, int] | None:
    """Map a Rajagopal muscle name to SMPL-24 joint indices (line endpoints)."""
    side = _muscle_side(name)
    low = name.lower()

    if any(x in low for x in ("rect_fem", "vas_", "semimem", "semiten", "bifem", "bflh", "bfsh", "iliacus", "psoas")):
        if side == "r":
            return (2, 5)
        if side == "l":
            return (1, 4)
    if "glut" in low:
        if side == "r":
            return (0, 2)
        if side == "l":
            return (0, 1)
    if any(x in low for x in ("gas", "soleus", "tib_", "per_")):
        if side == "r":
            return (5, 8)
        if side == "l":
            return (4, 7)
    if any(x in low for x in ("lumbar", "erec", "mult", "rect_abd")):
        return (0, 9)
    if "delt" in low:
        return (14, 17) if side == "r" else (13, 16) if side == "l" else (9, 16)
    if "pect" in low:
        return (9, 17) if side == "r" else (9, 16) if side == "l" else (9, 12)
    if "bic" in low or "tric" in low:
        if side == "r":
            return (17, 19)
        if side == "l":
            return (16, 18)
    return None


def _muscle_activation_by_joint_pair(
    frame_idx: int,
    activations: np.ndarray,
    muscle_names: list[str],
) -> dict[tuple[int, int], float]:
    """Max activation per undirected joint pair for drawing muscle lines."""
    t = int(np.clip(frame_idx, 0, activations.shape[0] - 1))
    pair_act: dict[tuple[int, int], float] = {}
    for i, name in enumerate(muscle_names):
        jp = _joint_pair_for_muscle(name)
        if jp is None:
            continue
        a, b = jp
        key = (a, b) if a < b else (b, a)
        val = float(activations[t, i])
        pair_act[key] = max(pair_act.get(key, 0.0), val)
    return pair_act


def get_smplx_skeleton_joints(smplx_frame: np.ndarray) -> np.ndarray:
    """FK from one SMPL-X 322-dim frame → 24 joint positions (Y-up)."""
    sl = _SMPLX_SLICES
    root_aa = smplx_frame[sl["root_orient"]].astype(np.float64, copy=False)
    body = smplx_frame[sl["pose_body"]].reshape(21, 3)
    trans = smplx_frame[sl["trans"]].astype(np.float64, copy=False)

    rotvec = np.zeros((24, 3), dtype=np.float64)
    rotvec[0] = root_aa
    rotvec[1:22] = body
    rotvec[22] = rotvec[20]
    rotvec[23] = rotvec[21]

    global_R: list[np.ndarray] = []
    for i in range(24):
        r_local = R.from_rotvec(rotvec[i]).as_matrix()
        p = int(_SMPL_PARENTS[i])
        global_R.append(r_local if p < 0 else global_R[p] @ r_local)

    joints = np.zeros((24, 3), dtype=np.float64)
    joints[0] = trans
    for i in range(1, 24):
        p = int(_SMPL_PARENTS[i])
        joints[i] = joints[p] + global_R[p] @ _SMPL_OFFSETS[i]
    return joints.astype(np.float32)


def _skeleton_segments() -> list[tuple[int, int]]:
    return [(int(_SMPL_PARENTS[i]), i) for i in range(24) if int(_SMPL_PARENTS[i]) >= 0]


def _draw_skeleton_frame(
    ax,
    joints: np.ndarray,
    activations: np.ndarray,
    muscle_names: list[str],
    t: int,
    *,
    segs: list[tuple[int, int]],
    cmap,
    bone_width: float,
    center: np.ndarray,
    radius: float,
    muscle_min_act: float = 0.04,
) -> None:
    """Gray kinematic skeleton + colored straight muscle lines between joints."""
    ax.cla()
    skel_color = "#6a6a6a"
    skel_width = max(1.0, bone_width * 0.45)

    for p, c in segs:
        ax.plot(
            [joints[p, 0], joints[c, 0]],
            [joints[p, 1], joints[c, 1]],
            [joints[p, 2], joints[c, 2]],
            color=skel_color,
            linewidth=skel_width,
            solid_capstyle="round",
            alpha=0.85,
            zorder=1,
        )

    muscle_width = max(2.5, bone_width * 1.15)
    for (a, b), act in _muscle_activation_by_joint_pair(t, activations, muscle_names).items():
        if act < muscle_min_act:
            continue
        rgba = cmap(float(np.clip(act, 0.0, 1.0)))
        ax.plot(
            [joints[a, 0], joints[b, 0]],
            [joints[a, 1], joints[b, 1]],
            [joints[a, 2], joints[b, 2]],
            color=rgba,
            linewidth=muscle_width,
            solid_capstyle="round",
            alpha=0.92,
            zorder=2,
        )

    ax.scatter(
        joints[:, 0],
        joints[:, 1],
        joints[:, 2],
        c="#f5f5f5",
        edgecolors="#222222",
        linewidths=0.6,
        s=22,
        depthshade=False,
        zorder=3,
    )
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_xlabel("X (right)")
    ax.set_ylabel("Y (up)")
    ax.set_zlabel("Z")
    ax.set_title(f"frame {t}", fontsize=10)
    ax.view_init(elev=18, azim=-65)
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass


def _coolwarm_cmap():
    try:
        return plt.colormaps["coolwarm"]
    except AttributeError:
        return cm.get_cmap("coolwarm")


def create_activation_skeleton_animation(
    smplx_motion: np.ndarray,
    activations: np.ndarray,
    muscle_names: list[str],
    *,
    fps: int = 12,
    frame_step: int = 1,
    title: str = "",
):
    """Build a matplotlib FuncAnimation for notebook display (use anim.to_jshtml())."""
    from matplotlib.animation import FuncAnimation

    motion = np.asarray(smplx_motion, dtype=np.float32)
    acts = np.asarray(activations, dtype=np.float32)
    n = min(int(motion.shape[0]), int(acts.shape[0]))
    if n < 1:
        raise ValueError("motion and activations must have at least one frame")
    step = max(1, int(frame_step))
    frames = list(range(0, n, step))

    segs = _skeleton_segments()
    cmap = _coolwarm_cmap()
    all_joints = np.stack(
        [get_smplx_skeleton_joints(motion[t]).astype(np.float64) for t in range(n)],
        axis=0,
    )
    lo = all_joints.reshape(-1, 3).min(axis=0)
    hi = all_joints.reshape(-1, 3).max(axis=0)
    center = 0.5 * (lo + hi)
    radius = float(np.max(hi - lo)) * 0.55 + 1e-3

    fig = plt.figure(figsize=(6.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    if title:
        fig.suptitle(title, fontsize=11, y=0.98)
    fig.subplots_adjust(left=0.02, right=0.92, top=0.90, bottom=0.08)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.08, label="Muscle activation")

    interval_ms = max(1, int(1000 / max(fps, 1)))

    def _update(i: int) -> None:
        t = frames[i]
        _draw_skeleton_frame(
            ax,
            all_joints[t],
            acts,
            muscle_names,
            t,
            segs=segs,
            cmap=cmap,
            bone_width=3.0,
            center=center,
            radius=radius,
        )

    anim = FuncAnimation(
        fig,
        _update,
        frames=len(frames),
        interval=interval_ms,
        repeat=True,
        blit=False,
    )
    anim._mm_fig = fig  # type: ignore[attr-defined]
    return anim


def create_triple_activation_animation(
    smplx_motion: np.ndarray,
    gt_acts: np.ndarray,
    mm_acts: np.ndarray,
    kin_acts: np.ndarray,
    muscle_names: list[str],
    *,
    fps: int = 12,
    frame_step: int = 1,
    title: str = "",
):
    """Side-by-side GT / MuscleMAP / Kinesis skeleton animation (one shared timeline)."""
    from matplotlib.animation import FuncAnimation

    motion = np.asarray(smplx_motion, dtype=np.float32)
    gt = np.asarray(gt_acts, dtype=np.float32)
    mm = np.asarray(mm_acts, dtype=np.float32)
    kin = np.asarray(kin_acts, dtype=np.float32)
    n = min(int(motion.shape[0]), int(gt.shape[0]), int(mm.shape[0]), int(kin.shape[0]))
    if n < 1:
        raise ValueError("motion and activations must have at least one frame")
    motion, gt, mm, kin = motion[:n], gt[:n], mm[:n], kin[:n]

    step = max(1, int(frame_step))
    frame_ids = list(range(0, n, step))
    segs = _skeleton_segments()
    cmap = _coolwarm_cmap()
    labels = ("Ground truth (OpenSim)", "MuscleMAP", "Kinesis (MyoLegs)")
    act_list = (gt, mm, kin)

    all_joints = np.stack(
        [get_smplx_skeleton_joints(motion[t]).astype(np.float64) for t in range(n)],
        axis=0,
    )
    lo = all_joints.reshape(-1, 3).min(axis=0)
    hi = all_joints.reshape(-1, 3).max(axis=0)
    center = 0.5 * (lo + hi)
    radius = float(np.max(hi - lo)) * 0.55 + 1e-3

    fig = plt.figure(figsize=(13.5, 4.8))
    axes = [fig.add_subplot(1, 3, i + 1, projection="3d") for i in range(3)]
    if title:
        fig.suptitle(title, fontsize=11, y=0.98)
    fig.subplots_adjust(left=0.02, right=0.90, top=0.88, bottom=0.08, wspace=0.12)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.04, label="Activation")

    interval_ms = max(1, int(1000 / max(fps, 1)))

    def _update(i: int) -> None:
        t = frame_ids[i]
        for ax, acts, label in zip(axes, act_list, labels):
            _draw_skeleton_frame(
                ax,
                all_joints[t],
                acts,
                muscle_names,
                t,
                segs=segs,
                cmap=cmap,
                bone_width=3.0,
                center=center,
                radius=radius,
            )
            ax.set_title(f"{label}\nframe {t}", fontsize=9)

    anim = FuncAnimation(
        fig,
        _update,
        frames=len(frame_ids),
        interval=interval_ms,
        repeat=True,
        blit=False,
    )
    anim._mm_fig = fig  # type: ignore[attr-defined]
    return anim


def save_animation_media(anim, path: str | Path, *, fps: int = 12) -> Path:
    """Save a FuncAnimation to MP4 (ffmpeg) or GIF (pillow) and return the written path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = getattr(anim, "_mm_fig", None)

    try:
        from matplotlib.animation import FFMpegWriter

        writer = FFMpegWriter(fps=fps, bitrate=1800)
        mp4 = out if out.suffix.lower() == ".mp4" else out.with_suffix(".mp4")
        anim.save(str(mp4), writer=writer, dpi=120)
        if fig is not None:
            plt.close(fig)
        return mp4
    except Exception as exc:
        from matplotlib.animation import PillowWriter

        gif = out.with_suffix(".gif")
        anim.save(str(gif), writer=PillowWriter(fps=fps))
        if fig is not None:
            plt.close(fig)
        if out.suffix.lower() == ".mp4":
            import warnings

            warnings.warn(f"MP4 export failed ({exc}); saved GIF instead: {gif}")
        return gif


def resolve_dataset_root(cfg: dict, train_cfg: dict) -> Path:
    """Resolve training dataset_root the same way as evaluate / benchmark loaders."""
    from src.loaders import _resolve_path

    raw = Path(str(train_cfg["data"]["dataset_root"]))
    if raw.is_absolute():
        return raw.resolve()
    model_repo = _resolve_path(cfg["paths"]["musclemap_model_repo"])
    via_model = (model_repo / raw).resolve()
    if via_model.is_dir():
        return via_model
    via_bench = _resolve_path(cfg["test_set"]["dataset_root"])
    if via_bench.is_dir():
        return via_bench
    return via_model


def find_sequence_dir(dataset_root: Path, sequence_id: str) -> Path:
    """Locate a sequence folder by id under the activations dataset root."""
    root = Path(dataset_root).expanduser()
    if not root.is_absolute():
        root = root.resolve()
    direct = root / sequence_id
    if direct.is_dir() and (direct / "smplx_322.npy").is_file():
        return direct
    matches = [p for p in root.rglob(sequence_id) if p.is_dir() and (p / "smplx_322.npy").is_file()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        return sorted(matches)[0]
    raise FileNotFoundError(f"Sequence {sequence_id!r} not found under {root}")


def load_best_example_arrays(
    best_ex: dict,
    cfg: dict,
    train_cfg: dict,
) -> tuple[np.ndarray, np.ndarray, list[str], Path]:
    """Load SMPL-X window, predicted activations, muscle names, and sequence dir."""
    seq_dir: Path | None = None
    if best_ex.get("sequence_dir"):
        candidate = Path(str(best_ex["sequence_dir"]))
        if candidate.is_dir() and (candidate / "smplx_322.npy").is_file():
            seq_dir = candidate.resolve()

    if seq_dir is None:
        dataset_root = resolve_dataset_root(cfg, train_cfg)
        seq_dir = find_sequence_dir(dataset_root, str(best_ex["sequence_id"]))

    smplx = np.load(seq_dir / "smplx_322.npy").astype(np.float32)
    start = int(best_ex.get("window_start", 0))
    true_t = int(best_ex["true_T"])
    smplx_win = smplx[start : start + true_t]
    pred = np.load(best_ex["pred_activations_npy"]).astype(np.float32)
    muscle_path = seq_dir.parent.parent / "muscle_names.json"
    if not muscle_path.is_file():
        dataset_root = resolve_dataset_root(cfg, train_cfg)
        muscle_path = dataset_root / train_cfg["data"]["muscle_names_json"]
    muscle_names = __import__("json").loads(muscle_path.read_text(encoding="utf-8"))
    return smplx_win, pred, muscle_names, seq_dir


def render_activation_skeleton_montage(
    smplx_motion: np.ndarray,
    activations: np.ndarray,
    muscle_names: list[str],
    out_path: Path,
    *,
    title: str = "",
    frame_indices: list[int] | None = None,
    bone_width: float = 2.5,
) -> Path:
    """Save a montage: human skeleton with bones colored by segment activation."""
    motion = np.asarray(smplx_motion, dtype=np.float32)
    acts = np.asarray(activations, dtype=np.float32)
    n = min(int(motion.shape[0]), int(acts.shape[0]))
    if n < 1:
        raise ValueError("motion and activations must have at least one frame")
    motion = motion[:n]
    acts = acts[:n]

    if frame_indices is None:
        frame_indices = [0, n // 2, n - 1] if n >= 3 else list(range(n))
    frame_indices = [int(np.clip(t, 0, n - 1)) for t in frame_indices]

    segs = _skeleton_segments()
    n_panels = len(frame_indices)
    fig = plt.figure(figsize=(4.2 * n_panels, 5.2))
    cmap = cm.get_cmap("coolwarm")
    axes_3d: list = []

    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for t in frame_indices:
        j = get_smplx_skeleton_joints(motion[t]).astype(np.float64)
        lo = np.minimum(lo, j.min(axis=0))
        hi = np.maximum(hi, j.max(axis=0))
    center = 0.5 * (lo + hi)
    radius = float(np.max(hi - lo)) * 0.55 + 1e-3

    for panel, t in enumerate(frame_indices):
        ax = fig.add_subplot(1, n_panels, panel + 1, projection="3d")
        axes_3d.append(ax)
        joints = get_smplx_skeleton_joints(motion[t]).astype(np.float64)
        _draw_skeleton_frame(
            ax,
            joints,
            acts,
            muscle_names,
            t,
            segs=segs,
            cmap=cmap,
            bone_width=bone_width,
            center=center,
            radius=radius,
        )

    if title:
        fig.suptitle(title, fontsize=11, y=0.98)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    # tight_layout() is unsupported for 3D axes; use manual margins + bbox_inches on save.
    fig.subplots_adjust(left=0.04, right=0.90, top=0.86, bottom=0.10, wspace=0.12)
    fig.colorbar(
        sm,
        ax=axes_3d,
        fraction=0.035,
        pad=0.04,
        label="Muscle activation",
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
