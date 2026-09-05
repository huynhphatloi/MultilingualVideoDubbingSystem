"""Speech generation providers."""
from __future__ import annotations

from typing import Optional

from core.runtime import slot
from providers.base import ModelSpec
from providers.tts.base import SpeechRequest, TTSProvider
from providers.tts.registry import ALIASES, DEFAULT_MODEL, REGISTRY

__all__ = [
    "REGISTRY", "DEFAULT_MODEL", "ALIASES", "SpeechRequest", "TTSProvider", "load",
]


def load(spec: ModelSpec, language: Optional[str] = None) -> TTSProvider:
    """The loaded engine. Language is part of the key for the per-language ones."""
    module = REGISTRY.implementation(spec)
    key = module.cache_key(spec, language)
    return slot("tts").get(key, lambda: module.build(spec, language=language))
