"""Model-cache locations that survive both the container and a native run.

The container image sets ``HF_HOME=/models/huggingface`` and friends, backed by
a Docker volume. Natively on macOS ``/models`` does not exist and the system
volume is read-only, so torch.hub / huggingface_hub die with::

    OSError: [Errno 30] Read-only file system: '/models'

...and they die *inside* whichever stage is first to download a checkpoint,
which makes it look like that model is broken rather than the cache path.

`ensure_model_cache()` resolves every cache variable to somewhere actually
writable, falling back to ``~/.cache/mvds/`` and saying so loudly.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

#: env var -> sub-directory used when the variable is unset or unusable.
_CACHE_VARS = {
    "HF_HOME": "huggingface",        # transformers, faster-whisper, pyannote
    "TORCH_HOME": "torch",           # torch.hub -> demucs checkpoints
    "XDG_CACHE_HOME": "cache",       # coqui-tts / XTTS
}

FALLBACK_ROOT = Path.home() / ".cache" / "mvds"

#: Hugging Face environment that has to be right *before* transformers is
#: imported. Values are only applied when the operator has not set them.
_HF_DEFAULTS = {
    # facebook/nllb-200-distilled-600M ships pytorch_model.bin and no
    # safetensors. After loading the .bin, transformers fires a background
    # "Thread-auto_conversion" that downloads the community safetensors PR -
    # a second, silent 2.46 GB for a model we already have in memory. On a
    # normal home connection that saturates the link for ~10 minutes, which is
    # why the *next* stage (MMS-TTS, 145 MB) appears to hang forever.
    # See transformers/modeling_utils.py, the Thread-auto_conversion block.
    "DISABLE_SAFETENSORS_CONVERSION": "true",
    # Tokenizers forks per request otherwise and spams the log with warnings.
    "TOKENIZERS_PARALLELISM": "false",
}


def _is_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write-probe"
        probe.touch()
        probe.unlink()
        return True
    except OSError:
        return False


def ensure_model_cache() -> dict[str, str]:
    """Point every cache env var at a writable absolute path. Returns the map."""
    resolved: dict[str, str] = {}

    for var, value in _HF_DEFAULTS.items():
        os.environ.setdefault(var, value)

    for var, subdir in _CACHE_VARS.items():
        raw = os.environ.get(var, "").strip()
        candidate = Path(raw).expanduser() if raw else FALLBACK_ROOT / subdir
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()

        if not _is_writable(candidate):
            fallback = FALLBACK_ROOT / subdir
            log.warning(
                "%s=%s is not writable - falling back to %s. "
                "Model downloads would otherwise fail mid-pipeline.",
                var, candidate, fallback,
            )
            candidate = fallback
            candidate.mkdir(parents=True, exist_ok=True)

        os.environ[var] = str(candidate)
        resolved[var] = str(candidate)

    return resolved


def cache_report() -> dict[str, dict]:
    """Used by /health/ready so a bad cache path is visible before a job runs."""
    report: dict[str, dict] = {}
    for var in _CACHE_VARS:
        raw = os.environ.get(var)
        path = Path(raw) if raw else None
        report[var] = {
            "path": raw,
            "exists": bool(path and path.exists()),
            "writable": bool(path and _is_writable(path)),
        }
    return report
