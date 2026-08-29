"""Device selection for CPU / CUDA / Apple Metal (MPS).

Not every backend speaks every device, which is why this module resolves a
device *per component* instead of globally:

============  ====  ====  ==================================================
component     cuda  mps   note
============  ====  ====  ==================================================
asr           yes   NO    faster-whisper runs on CTranslate2, which has no
                          Metal backend at all -> silently mapped to cpu.
                          On Apple Silicon the int8 CPU path uses NEON and is
                          still respectable.
separation    yes   yes   Demucs is plain torch.
diarization   yes   yes   pyannote is torch, but a few ops have no Metal
                          kernel -> needs PYTORCH_ENABLE_MPS_FALLBACK=1.
translation   yes   yes   transformers seq2seq.
tts           yes   yes   XTTS / VITS are plain torch.
============  ====  ====  ==================================================

`DEVICE=auto` picks cuda > mps > cpu. Each component can be pinned
individually (`ASR_DEVICE`, `TTS_DEVICE`, ...) when one of them misbehaves.
"""
from __future__ import annotations

import contextlib
import logging
import os
from functools import lru_cache

from app.core.config import settings

log = logging.getLogger(__name__)

#: Components whose backend cannot use Metal, whatever the user asks for.
_NO_MPS = {"asr"}

_COMPONENTS = ("asr", "separation", "diarization", "translation", "tts")


# --------------------------------------------------------------- detection ---
@lru_cache(maxsize=1)
def _probe() -> dict[str, bool]:
    """What the installed torch can actually do."""
    caps = {"cuda": False, "mps": False}
    try:
        import torch
    except Exception as exc:  # pragma: no cover - torch missing/broken
        log.warning("torch unavailable, falling back to cpu: %s", exc)
        return caps

    with contextlib.suppress(Exception):  # pragma: no cover
        caps["cuda"] = bool(torch.cuda.is_available())
    with contextlib.suppress(Exception):  # pragma: no cover
        backend = getattr(torch.backends, "mps", None)
        caps["mps"] = bool(backend and backend.is_available() and backend.is_built())
    return caps


@lru_cache(maxsize=1)
def resolve_device() -> str:
    """The default device for the whole service."""
    requested = (settings.device or "auto").lower()
    caps = _probe()

    if requested == "cpu":
        return "cpu"

    if requested in ("cuda", "mps"):
        if caps[requested]:
            return requested
        log.warning("DEVICE=%s requested but not available -> using cpu", requested)
        return "cpu"

    # auto
    if caps["cuda"]:
        return "cuda"
    if caps["mps"]:
        return "mps"
    return "cpu"


def resolve_device_for(component: str) -> str:
    """Device for one pipeline component, honouring per-component overrides."""
    if component not in _COMPONENTS:
        raise ValueError(f"Unknown component '{component}'")

    override = (getattr(settings, f"{component}_device", "auto") or "auto").lower()
    device = resolve_device() if override == "auto" else override

    if device != "cpu" and not _probe().get(device, False):
        log.warning("%s: device '%s' unavailable -> cpu", component, device)
        device = "cpu"

    if device == "mps" and component in _NO_MPS:
        log.debug("%s has no Metal backend -> running on cpu", component)
        return "cpu"

    return device


@lru_cache(maxsize=1)
def resolve_compute_type() -> str:
    """CTranslate2 compute type used by faster-whisper."""
    if settings.compute_type != "auto":
        return settings.compute_type
    return "float16" if resolve_device_for("asr") == "cuda" else "int8"




# ----------------------------------------------------------------- runtime ---
def configure_threads() -> None:
    """Keep CPU inference from oversubscribing, and make Metal forgiving."""
    n = max(1, settings.torch_num_threads)
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    os.environ.setdefault("MKL_NUM_THREADS", str(n))

    # Several pyannote / coqui ops have no Metal kernel. Without this flag the
    # process raises NotImplementedError instead of quietly using the CPU.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    with contextlib.suppress(Exception):  # pragma: no cover
        import torch

        torch.set_num_threads(n)


def device_report() -> dict:
    caps = _probe()
    info: dict = {
        "requested": settings.device,
        "resolved": resolve_device(),
        "available": dict(caps),
        "per_component": {c: resolve_device_for(c) for c in _COMPONENTS},
        "compute_type": resolve_compute_type(),
        "mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
    }
    try:
        import torch

        info["torch_version"] = torch.__version__
        if caps["cuda"]:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
        elif caps["mps"]:
            info["gpu_name"] = "Apple Silicon GPU (Metal)"
    except Exception as exc:
        info["torch_error"] = str(exc)
    return info
