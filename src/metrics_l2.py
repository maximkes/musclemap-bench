from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.linalg import sqrtm

from src.loaders import _resolve_path, load_config

logger = logging.getLogger(__name__)

_MOTION_FEAT_DIM = 263
_T2M_BUNDLE: "_T2MEvaluator | None" = None


@dataclass
class _T2MEvaluator:
    """Loaded TM2T encoders and HumanML3D normalization stats."""

    t2m_textencoder: Any
    t2m_moveencoder: Any
    t2m_motionencoder: Any
    w_vectorizer: Any
    mean: np.ndarray
    std: np.ndarray
    unit_len: int
    max_text_len: int
    device: str


def _motiongpt_vendor_root(cfg: dict[str, Any]) -> Path:
    """Return the vendored MotionGPT root directory."""
    return (_resolve_path(cfg["paths"]["musclemap_model_repo"]) / "vendor" / "MotionGPT").resolve()


def _inject_motiongpt(vendor_root: Path) -> None:
    """Add vendored MotionGPT to sys.path for mGPT imports."""
    motiongpt_dir = str(vendor_root)
    if motiongpt_dir not in sys.path:
        sys.path.insert(0, motiongpt_dir)


def _stub_word_vectorizer() -> Any:
    """Minimal WordVectorizer when GloVe assets are not installed."""
    pos_size = 15

    class _StubWordVectorizer:
        def __getitem__(self, token: str) -> tuple[np.ndarray, np.ndarray]:
            _word, pos = token.split("/")
            pos_vec = np.zeros(pos_size, dtype=np.float32)
            pos_vec[0] = 1.0
            return np.zeros(300, dtype=np.float32), pos_vec

    return _StubWordVectorizer()


def _load_word_vectorizer(vendor_root: Path, dataset_cfg: Any) -> Any:
    """Load HumanML3D WordVectorizer or fall back to a stub."""
    from mGPT.data.humanml.utils.word_vectorizer import WordVectorizer  # type: ignore[import-not-found]

    glove_root = vendor_root / "deps" / "glove"
    prefix = "our_vab"
    if (glove_root / f"{prefix}_data.npy").exists():
        return WordVectorizer(str(glove_root), prefix)
    logger.warning("GloVe assets missing at %s; using stub word vectors.", glove_root)
    return _stub_word_vectorizer()


def _build_t2m_metric_cfg(train_cfg: dict[str, Any], vendor_root: Path) -> Any:
    """Build OmegaConf METRIC/DATASET blocks for TM2T evaluator loading."""
    from omegaconf import OmegaConf  # type: ignore[import-not-found]

    mean_std_root = vendor_root / "deps" / "t2m" / "t2m"
    metric_block = train_cfg.get("METRIC", {})
    if not metric_block:
        raise KeyError("musclemap train config must define METRIC.TM2T encoder targets")
    metric_cfg = OmegaConf.create(metric_block)
    metric_cfg.TM2T.t2m_path = str(mean_std_root)
    dataset_cfg = OmegaConf.create({
        "HUMANML3D": {
            "UNIT_LEN": 4,
            "MAX_TEXT_LEN": 20,
            "MEAN_STD_PATH": str(mean_std_root),
        },
    })
    return metric_cfg, dataset_cfg


def _load_evaluator(cfg: dict[str, Any]) -> _T2MEvaluator:
    """Load and cache TM2T text/motion encoders from the vendored MotionGPT layout."""
    global _T2M_BUNDLE
    if _T2M_BUNDLE is not None:
        return _T2M_BUNDLE

    import torch
    from mGPT.config import instantiate_from_config  # type: ignore[import-not-found]

    vendor_root = _motiongpt_vendor_root(cfg)
    if not vendor_root.is_dir():
        raise FileNotFoundError(f"MotionGPT vendor not found at {vendor_root}")

    _inject_motiongpt(vendor_root)
    train_cfg = load_config(_resolve_path(cfg["paths"]["musclemap_train_config"]))
    metric_cfg, dataset_cfg = _build_t2m_metric_cfg(train_cfg, vendor_root)

    finest = metric_cfg.TM2T.t2m_path + "/t2m/text_mot_match/model/finest.tar"
    finest_path = Path(str(finest)).resolve()
    if not finest_path.is_file():
        raise FileNotFoundError(f"TM2T evaluator checkpoint not found: {finest_path}")

    meta_dir = Path(str(metric_cfg.TM2T.t2m_path)).resolve() / "t2m" / "Comp_v6_KLD01" / "meta"
    mean_path = meta_dir / "mean.npy"
    std_path = meta_dir / "std.npy"
    if not mean_path.is_file() or not std_path.is_file():
        raise FileNotFoundError(f"HumanML3D mean/std not found under {meta_dir}")

    device = str(cfg.get("inference", {}).get("device", "cpu"))
    text_encoder = instantiate_from_config(metric_cfg.TM2T.t2m_textencoder)
    move_encoder = instantiate_from_config(metric_cfg.TM2T.t2m_moveencoder)
    motion_encoder = instantiate_from_config(metric_cfg.TM2T.t2m_motionencoder)

    checkpoint = torch.load(str(finest_path), map_location="cpu", weights_only=False)
    text_encoder.load_state_dict(checkpoint["text_encoder"])
    move_encoder.load_state_dict(checkpoint["movement_encoder"])
    motion_encoder.load_state_dict(checkpoint["motion_encoder"])

    for module in (text_encoder, move_encoder, motion_encoder):
        module.eval()
        module.to(device)
        for param in module.parameters():
            param.requires_grad = False

    w_vectorizer = _load_word_vectorizer(vendor_root, dataset_cfg)
    unit_len = int(dataset_cfg.HUMANML3D.UNIT_LEN)
    max_text_len = int(dataset_cfg.HUMANML3D.MAX_TEXT_LEN)

    _T2M_BUNDLE = _T2MEvaluator(
        t2m_textencoder=text_encoder,
        t2m_moveencoder=move_encoder,
        t2m_motionencoder=motion_encoder,
        w_vectorizer=w_vectorizer,
        mean=np.load(str(mean_path)).astype(np.float32),
        std=np.load(str(std_path)).astype(np.float32),
        unit_len=unit_len,
        max_text_len=max_text_len,
        device=device,
    )
    logger.info("Loaded TM2T evaluator from %s", finest_path)
    return _T2M_BUNDLE


def _get_nlp() -> Any | None:
    """Return a spaCy English model when available."""
    try:
        import spacy  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return None
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        try:
            return spacy.load("en")
        except OSError:
            return None


def _process_text(sentence: str, nlp: Any | None) -> tuple[list[str], list[str]]:
    """Tokenize text into words and POS tags for TM2T encoding."""
    sentence = sentence.replace("-", "")
    if nlp is not None:
        doc = nlp(sentence)
        word_list: list[str] = []
        pos_list: list[str] = []
        for token in doc:
            word = token.text
            if not word.isalpha():
                continue
            if (token.pos_ == "NOUN" or token.pos_ == "VERB") and word != "left":
                word_list.append(token.lemma_)
            else:
                word_list.append(word)
            pos_list.append(token.pos_)
        return word_list, pos_list

    words = [w for w in sentence.lower().split() if w.isalpha()]
    return words, ["OTHER"] * len(words)


def _encode_text_batch(texts: list[str], ev: _T2MEvaluator) -> np.ndarray:
    """Encode raw captions into TM2T text embeddings."""
    import torch

    nlp = _get_nlp()
    word_embs: list[torch.Tensor] = []
    pos_ohot: list[torch.Tensor] = []
    text_lengths: list[int] = []

    for sentence in texts:
        word_list, pos_list = _process_text(sentence.strip(), nlp)
        t_tokens = [f"{word_list[i]}/{pos_list[i]}" for i in range(len(word_list))]

        if len(t_tokens) < ev.max_text_len:
            tokens = ["sos/OTHER"] + t_tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"] * (ev.max_text_len + 2 - sent_len)
        else:
            tokens = ["sos/OTHER"] + t_tokens[: ev.max_text_len] + ["eos/OTHER"]
            sent_len = len(tokens)

        pos_one_hots: list[torch.Tensor] = []
        word_embeddings: list[torch.Tensor] = []
        for token in tokens:
            word_emb, pos_oh = ev.w_vectorizer[token]
            pos_one_hots.append(torch.tensor(pos_oh, dtype=torch.float32).unsqueeze(0))
            word_embeddings.append(torch.tensor(word_emb, dtype=torch.float32).unsqueeze(0))
        text_lengths.append(sent_len)
        pos_ohot.append(torch.cat(pos_one_hots, dim=0).unsqueeze(0))
        word_embs.append(torch.cat(word_embeddings, dim=0).unsqueeze(0))

    word_embs_t = torch.cat(word_embs, dim=0).to(ev.device)
    pos_ohot_t = torch.cat(pos_ohot, dim=0).to(ev.device)
    text_lengths_t = torch.tensor(text_lengths, device=ev.device)

    with torch.no_grad():
        text_emb = ev.t2m_textencoder(word_embs_t, pos_ohot_t, text_lengths_t)
    return torch.flatten(text_emb, start_dim=1).detach().cpu().numpy().astype(np.float32, copy=False)


def _encode_motion_batch(motions: list[np.ndarray], ev: _T2MEvaluator) -> np.ndarray:
    """Encode motion arrays into TM2T motion embeddings."""
    import torch

    out: list[np.ndarray] = []
    for motion in motions:
        feats = np.asarray(motion, dtype=np.float32)
        if feats.ndim != 2 or feats.shape[1] != _MOTION_FEAT_DIM:
            raise ValueError(f"Each motion must be [T, {_MOTION_FEAT_DIM}], got {feats.shape}")
        normed = (feats - ev.mean) / ev.std
        tensor = torch.from_numpy(normed).unsqueeze(0).to(ev.device)
        length = int(feats.shape[0])
        with torch.no_grad():
            m_lens = torch.tensor([length], device=ev.device)
            m_lens = torch.div(m_lens, ev.unit_len, rounding_mode="floor")
            m_lens = m_lens // ev.unit_len
            mov = ev.t2m_moveencoder(tensor[..., :-4]).detach()
            emb = ev.t2m_motionencoder(mov, m_lens)
            flat = torch.flatten(emb, start_dim=1).detach().cpu().numpy().astype(np.float32, copy=False)
        out.append(flat[0])
    return np.stack(out, axis=0).astype(np.float32, copy=False)


def extract_motion_features(motions: list[np.ndarray], cfg: dict[str, Any]) -> np.ndarray:
    """Extract TM2T motion embeddings for a list of [T, 263] motions."""
    ev = _load_evaluator(cfg)
    if not motions:
        return np.zeros((0, 0), dtype=np.float32)
    return _encode_motion_batch(motions, ev)


def extract_text_features(texts: list[str], cfg: dict[str, Any]) -> np.ndarray:
    """Extract TM2T text embeddings for a list of captions."""
    ev = _load_evaluator(cfg)
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    return _encode_text_batch(texts, ev)


def compute_fid(real_feats: np.ndarray, fake_feats: np.ndarray) -> float:
    """Compute Fréchet distance between real and generated feature sets."""
    mu_r, mu_f = real_feats.mean(0), fake_feats.mean(0)
    cov_r = np.cov(real_feats, rowvar=False)
    cov_f = np.cov(fake_feats, rowvar=False)
    diff = mu_r - mu_f
    covmean, _ = sqrtm(cov_r @ cov_f, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(cov_r + cov_f - 2 * covmean))


def compute_r_precision(motion_feats: np.ndarray, text_feats: np.ndarray, top_k: Sequence[int] = (1, 2, 3)) -> dict[str, float]:
    """Compute R-precision at the given top-k values."""
    B = motion_feats.shape[0]
    res: dict[str, float] = {}
    for k in top_k:
        hits = 0
        for i in range(B):
            dists = np.linalg.norm(text_feats - motion_feats[i], axis=1)
            ranked = np.argsort(dists)
            if i in ranked[:k]:
                hits += 1
        res[f"r_precision_top{k}"] = hits / B
    return res


def compute_mm_dist(motion_feats: np.ndarray, text_feats: np.ndarray) -> float:
    """Compute mean Euclidean distance between paired motion/text features."""
    return float(np.linalg.norm(motion_feats - text_feats, axis=1).mean())


def compute_diversity(motion_feats: np.ndarray, n_pairs: int = 300) -> float:
    """Compute average pairwise distance across random motion feature pairs."""
    B = motion_feats.shape[0]
    rng = np.random.default_rng(42)
    idx = rng.choice(B, size=(n_pairs, 2), replace=True)
    dists = np.linalg.norm(motion_feats[idx[:, 0]] - motion_feats[idx[:, 1]], axis=1)
    return float(dists.mean())


def compute_l2_metrics(motions: list[np.ndarray], texts: list[str], cfg: dict[str, Any], *, real_motions: list[np.ndarray] | None = None) -> dict[str, Any]:
    """Compute Layer 2 HumanML3D metrics (FID, R-precision, MM-dist, diversity)."""
    motion_feats = extract_motion_features(motions, cfg)
    text_feats = extract_text_features(texts, cfg)
    real_feats = extract_motion_features(real_motions, cfg) if real_motions is not None else motion_feats
    rng = np.random.default_rng(0)
    fid_vals: list[float] = []
    for _ in range(cfg["layer2"]["n_replication"]):
        n = min(len(motion_feats), 256)
        idx = rng.choice(len(motion_feats), size=n, replace=False)
        fid_vals.append(compute_fid(real_feats[idx], motion_feats[idx]))
    return {
        "fid_mean": float(np.mean(fid_vals)),
        "fid_std": float(np.std(fid_vals)),
        **compute_r_precision(motion_feats, text_feats, top_k=cfg["layer2"]["top_k"]),
        "mm_dist": compute_mm_dist(motion_feats, text_feats),
        "diversity": compute_diversity(motion_feats),
    }


def reset_evaluator_cache() -> None:
    """Clear the cached TM2T evaluator (for tests)."""
    global _T2M_BUNDLE
    _T2M_BUNDLE = None
