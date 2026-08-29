"""Lazy, process-wide singletons for the heavy models.

n8n calls each stage as a separate HTTP request, so keeping the models warm
between calls is what makes the pipeline usable. Loading is guarded by a lock
so two concurrent jobs never load the same model twice.
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

log = logging.getLogger(__name__)

_registry: dict[str, Any] = {}
_meta: dict[str, dict] = {}
_locks: dict[str, threading.Lock] = {}
_global_lock = threading.Lock()


def _lock_for(key: str) -> threading.Lock:
    with _global_lock:
        return _locks.setdefault(key, threading.Lock())


def get_or_load(key: str, loader: Callable[[], Any]) -> Any:
    """Return a cached model, loading it (once) if needed."""
    if key in _registry:
        _meta[key]["last_used"] = time.time()
        _meta[key]["hits"] += 1
        return _registry[key]

    with _lock_for(key):
        if key in _registry:  # another thread won the race
            return _registry[key]
        started = time.perf_counter()
        log.info("loading model '%s' ...", key)
        obj = loader()
        elapsed = time.perf_counter() - started
        _registry[key] = obj
        _meta[key] = {"loaded_at": time.time(), "load_seconds": round(elapsed, 2),
                      "last_used": time.time(), "hits": 1}
        log.info("model '%s' ready in %.1fs", key, elapsed)
        return obj


def unload(key: str) -> bool:
    with _lock_for(key):
        if key not in _registry:
            return False
        del _registry[key]
        _meta.pop(key, None)
    _free_memory()
    return True


def unload_all() -> int:
    keys = list(_registry)
    for key in keys:
        unload(key)
    return len(keys)


def loaded() -> dict[str, dict]:
    return {k: dict(v) for k, v in _meta.items()}


def _free_memory() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover
        pass
