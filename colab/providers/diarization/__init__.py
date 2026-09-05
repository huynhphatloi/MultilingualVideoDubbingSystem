"""Diarization providers."""
from __future__ import annotations

from core.runtime import slot
from providers.base import ModelSpec
from providers.diarization.base import DiarizationProvider
from providers.diarization.registry import DEFAULT_MODEL, REGISTRY

__all__ = ["REGISTRY", "DEFAULT_MODEL", "DiarizationProvider", "load"]


def load(spec: ModelSpec) -> DiarizationProvider:
    module = REGISTRY.implementation(spec)
    return slot("diarization").get(module.cache_key(spec), lambda: module.build(spec))
