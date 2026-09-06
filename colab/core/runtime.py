"""Device selection, optional-package probing and the model slots.

Colab's free GPU cannot hold two large checkpoints, so each task owns a slot
that holds exactly one loaded model. Asking a slot for a different key evicts
what it holds first. The registry is metadata; the slots are the only place a
loaded model lives.
"""
from __future__ import annotations

import gc
import importlib.util
import os
import threading
from typing import Callable, Dict, Optional

from .errors import MissingDependency

_device: Optional[str] = None


def device() -> str:
    """"cuda" when torch sees a GPU. torch is imported lazily so that the HTTP
    layer, the registry and the tests all work without it."""
    global _device
    if _device is None:
        forced = os.getenv("DUBFLOW_DEVICE", "").strip().lower()
        if forced in {"cuda", "cpu"}:
            _device = forced
        else:
            try:
                import torch

                _device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:  # noqa: BLE001 - no torch is a valid state here
                _device = "cpu"
    return _device


def free_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def installed(import_name: Optional[str]) -> bool:
    """Is this optional dependency importable in this session?"""
    if not import_name:
        return True
    try:
        return importlib.util.find_spec(import_name) is not None
    except (ImportError, ValueError):
        return False


def require(import_name: str, pip_name: str, purpose: str) -> None:
    if not installed(import_name):
        raise MissingDependency(pip_name, purpose)


class Slot:
    """One loaded model per slot, evicted when the key changes.

    The key identifies everything that changes the loaded object - the model id
    plus whatever else the builder used, such as an MMS language adapter.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._key: Optional[str] = None
        self._value = None
        self._lock = threading.Lock()

    @property
    def key(self) -> Optional[str]:
        return self._key

    def get(self, key: str, build: Callable[[], object]):  # noqa: ANN201
        with self._lock:
            if self._key == key and self._value is not None:
                return self._value
            self.release()
            value = build()
            self._value = value
            self._key = key
            return value

    def release(self) -> None:
        if self._value is None and self._key is None:
            return
        self._value = None
        self._key = None
        free_memory()


#: One slot per task. Two tasks may hold a model at once (ASR and translation
#: never run at the same moment, but the queue keeps whichever ran last), which
#: is the deliberate trade: re-loading Whisper between every stage would cost
#: more than the memory it frees.
SLOTS: Dict[str, Slot] = {
    task: Slot(task)
    for task in ("asr", "translation", "tts", "diarization", "separation")
}


def slot(task: str) -> Slot:
    if task not in SLOTS:
        SLOTS[task] = Slot(task)
    return SLOTS[task]


def release_all() -> None:
    for entry in SLOTS.values():
        entry.release()


def loaded() -> Dict[str, Optional[str]]:
    return {name: entry.key for name, entry in SLOTS.items()}
