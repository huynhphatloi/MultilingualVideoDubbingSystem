"""Source separation providers."""
from __future__ import annotations

from core.runtime import slot
from providers.base import ModelSpec
from providers.separation.base import SourceSeparationProvider
from providers.separation.registry import DEFAULT_MODEL, REGISTRY

__all__ = ["REGISTRY", "DEFAULT_MODEL", "SourceSeparationProvider", "load"]


def load(spec: ModelSpec) -> SourceSeparationProvider:
    module = REGISTRY.implementation(spec)
    return slot("separation").get(module.cache_key(spec), lambda: module.build(spec))
