"""Translation providers."""
from __future__ import annotations

from core.runtime import slot
from providers.base import ModelSpec
from providers.translation.base import TranslationProvider
from providers.translation.registry import DEFAULT_MODEL, REGISTRY

__all__ = ["REGISTRY", "DEFAULT_MODEL", "TranslationProvider", "load"]


def load(spec: ModelSpec) -> TranslationProvider:
    module = REGISTRY.implementation(spec)
    return slot("translation").get(module.cache_key(spec), lambda: module.build(spec))
