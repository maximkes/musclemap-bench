from __future__ import annotations

import contextlib
import io
import logging
import time
import types
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

HUMANML3D_MAX_FRAMES = 196


@dataclass
class BenchSample:
    """One model inference result for benchmarking."""

    sequence_id: str
    text: str
    activations: np.ndarray | None = None
    motion: np.ndarray | None = None
    timing_s: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


def _ensure_teacher_forced_backbone(model: Any) -> None:
    """Patch MotionGPT backbone for a single teacher-forced T5 pass (matches evaluate.py)."""
    import torch

    backbone = model.backbone
    if getattr(backbone, "_musclemap_bench_eval_patch", False):
        return

    def _teacher_forced_forward(
        self: Any, batch: dict[str, Any], task: str = "t2m"
    ) -> dict[str, torch.Tensor]:
        del task
        texts = batch["text"]
        if not isinstance(texts, list):
            texts = [str(t) for t in texts]

        lm = getattr(self, "lm", None)
        language_model = getattr(lm, "language_model", None) if lm is not None else None
        tokenizer = getattr(lm, "tokenizer", None) if lm is not None else None
        if language_model is None or tokenizer is None:
            raise RuntimeError("MotionGPT backbone missing lm.language_model or lm.tokenizer")

        device = language_model.device
        enc = tokenizer(
            texts,
            padding="max_length",
            max_length=int(getattr(lm, "max_length", 256)),
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        start_id = int(
            getattr(getattr(language_model, "config", None), "decoder_start_token_id", None)
            or getattr(getattr(language_model, "config", None), "pad_token_id", 0)
        )
        decoder_input_ids = torch.full(
            (input_ids.shape[0], 1),
            fill_value=start_id,
            device=device,
            dtype=input_ids.dtype,
        )
        out = language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        enc_h = getattr(out, "encoder_last_hidden_state", None)
        dec_h = getattr(out, "decoder_last_hidden_state", None)
        if dec_h is None:
            dec_states = getattr(out, "decoder_hidden_states", None)
            if dec_states:
                dec_h = dec_states[-1]
        if enc_h is None or dec_h is None:
            raise RuntimeError("Teacher-forced T5 forward did not return hidden states")
        return {"encoder_hidden": enc_h, "decoder_hidden": dec_h}

    backbone.forward = types.MethodType(_teacher_forced_forward, backbone)  # type: ignore[method-assign]
    backbone._musclemap_bench_eval_patch = True
    logger.debug("Patched MotionGPT backbone for teacher-forced MuscleMAP inference.")


def _predicted_length(logits: Any, pred_log_T: Any, config: dict[str, Any] | None) -> int:
    """Return the number of activation frames produced by the model."""
    import torch

    t_logits = int(logits.shape[1])
    if config is None:
        return t_logits
    lp_cfg = config.get("model", {}).get("length_predictor", {})
    min_t = int(lp_cfg.get("min_T", 30))
    max_t = int(lp_cfg.get("max_T", 256))
    pred_t = torch.exp(pred_log_T).round().clamp(min=float(min_t), max=float(max_t)).to(dtype=torch.int64)
    return int(min(t_logits, int(pred_t[0].item())))


def _motion_from_output(motion_output: Any) -> np.ndarray | None:
    """Extract HumanML3D motion features from a MotionGPT forward return value."""
    import torch

    if not isinstance(motion_output, dict):
        return None
    feats = motion_output.get("feats")
    if feats is None or not isinstance(feats, torch.Tensor):
        return None
    lengths = motion_output.get("length")
    if isinstance(lengths, (list, tuple)) and lengths:
        t_len = int(lengths[0])
        return feats[0, :t_len].detach().cpu().numpy().astype(np.float32, copy=False)
    if feats.ndim == 3 and feats.shape[0] == 1:
        return feats[0].detach().cpu().numpy().astype(np.float32, copy=False)
    return None


def run_musclemap(
    model: Any,
    text: str,
    sequence_id: str,
    device: str = "cpu",
    *,
    ref_T: int | None = None,
) -> BenchSample:
    """Run MuscleMAP text-to-activation inference for one benchmark sample."""
    import torch

    model.to(device)
    model.eval()
    _ensure_teacher_forced_backbone(model)

    forward_kwargs: dict[str, Any] = {
        "text_tokens": [text],
        "motion_tokens": None,
    }
    if ref_T is not None and ref_T > 0:
        forward_kwargs["T_frame"] = int(ref_T)

    t0 = time.perf_counter()
    with torch.no_grad():
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            logits, pred_log_T, motion_output = model(**forward_kwargs)
    t1 = time.perf_counter()

    probs = torch.sigmoid(logits)[0].detach().cpu().numpy().astype(np.float32, copy=False)
    if ref_T is not None and ref_T > 0:
        # Ground-truth length is known at benchmark/eval time; do not truncate via length predictor.
        pred_t = min(probs.shape[0], int(ref_T))
    else:
        pred_t = _predicted_length(logits, pred_log_T, getattr(model, "config", None))
    probs = probs[:pred_t]

    motion_np = _motion_from_output(motion_output)
    if motion_np is not None and ref_T is not None and ref_T > 0:
        motion_np = motion_np[: min(motion_np.shape[0], int(ref_T))]

    return BenchSample(
        sequence_id=sequence_id,
        text=text,
        activations=probs,
        motion=motion_np,
        timing_s=t1 - t0,
        meta={"pred_T": pred_t},
    )


def run_motiongpt(
    backbone: Any,
    text: str,
    sequence_id: str,
    device: str = "cpu",
    *,
    ref_T: int | None = None,
    max_frames: int = HUMANML3D_MAX_FRAMES,
) -> BenchSample:
    """Run MotionGPT text-to-motion generation and return HumanML3D features [T, 263]."""
    if getattr(backbone, "_musclemap_bench_eval_patch", False):
        raise RuntimeError(
            "MotionGPT backbone was patched for MuscleMAP eval; load a fresh backbone via load_motiongpt_backbone()."
        )

    import torch

    backbone.to(device)
    backbone.eval()

    t0 = time.perf_counter()
    with torch.no_grad():
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            motion_output = backbone.forward(
                {"text": [text], "length": [int(max_frames)]},
                task="t2m",
            )
    t1 = time.perf_counter()

    motion_np = _motion_from_output(motion_output)
    if motion_np is None:
        raise RuntimeError("MotionGPT forward did not return 'feats' and 'length'")
    if motion_np.ndim != 2 or motion_np.shape[1] != 263:
        raise ValueError(f"Expected motion [T, 263], got {motion_np.shape}")

    gen_len = int(motion_output.get("length", [motion_np.shape[0]])[0]) if isinstance(motion_output, dict) else motion_np.shape[0]
    if ref_T is not None and ref_T > 0:
        motion_np = motion_np[: min(motion_np.shape[0], int(ref_T))]

    return BenchSample(
        sequence_id=sequence_id,
        text=text,
        activations=None,
        motion=motion_np.astype(np.float32, copy=False),
        timing_s=t1 - t0,
        meta={"task": "t2m", "gen_T": gen_len},
    )


def run_kinesis_from_artifact(npy_path: str, sequence_id: str, text: str) -> BenchSample:
    """Load precomputed Kinesis activations from disk."""
    arr = np.load(npy_path).astype(np.float32)
    return BenchSample(
        sequence_id=sequence_id,
        text=text,
        activations=arr,
        timing_s=float("nan"),
        meta={"source": "kinesis_artifact"},
    )
