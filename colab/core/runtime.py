"""Device selection, dependency probing, and model slots."""
from __future__ import annotations

import gc
import importlib.util
import os
import threading
from typing import Callable, Dict, Optional

_device: Optional[str] = None


def device() -> str:
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
    if not import_name:
        return True
    try:
        return importlib.util.find_spec(import_name) is not None
    except (ImportError, ValueError):
        return False


class Slot:
    """One loaded model per task, evicted when its key changes."""

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
