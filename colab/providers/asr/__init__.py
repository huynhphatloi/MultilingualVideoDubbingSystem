"""Speech recognition providers."""
from __future__ import annotations

from typing import Optional

from core.runtime import slot
from providers.asr.base import ASRProvider, Transcript, Window, windows_from_turns
from providers.asr.registry import DEFAULT_MODEL, REGISTRY
from providers.base import ModelSpec

__all__ = [
    "REGISTRY", "DEFAULT_MODEL", "ASRProvider", "Transcript", "Window",
    "windows_from_turns", "load",
]


def load(spec: ModelSpec, language: Optional[str] = None) -> ASRProvider:
    """The loaded recogniser, reusing the resident one when the key matches."""
    module = REGISTRY.implementation(spec)
    key = module.cache_key(spec, language)
    return slot("asr").get(key, lambda: module.build(spec, language))
