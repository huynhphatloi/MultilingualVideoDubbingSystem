"""Lip-sync providers."""
from __future__ import annotations

from core.runtime import slot
from providers.base import ModelSpec
from providers.lipsync.base import LipSyncProvider
from providers.lipsync.registry import DEFAULT_MODEL, REGISTRY

__all__ = ["REGISTRY", "DEFAULT_MODEL", "LipSyncProvider", "load"]


def load(spec: ModelSpec) -> LipSyncProvider:
    module = REGISTRY.implementation(spec)
    return slot("lipsync").get(module.cache_key(spec), lambda: module.build(spec))
