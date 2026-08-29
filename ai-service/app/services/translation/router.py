"""Engine selection + automatic fallback.

n8n can also branch on failure explicitly, but having the fallback here means
a single HTTP call never fails just because the primary model choked.
"""
from __future__ import annotations

import logging

from app.core.config import settings
from app.core.errors import TranslationError
from app.services.translation.base import TranslationEngine
from app.services.translation.nllb import NLLBEngine
from app.services.translation.remote import RemoteTranslationEngine
from app.services.translation.seamless import SeamlessEngine

log = logging.getLogger(__name__)

_BUILDERS = {
    "remote": RemoteTranslationEngine,
    "nllb": NLLBEngine,
    "seamless": SeamlessEngine,
}


def build(name: str) -> TranslationEngine:
    if name not in _BUILDERS:
        raise TranslationError(f"Unknown translation engine '{name}'",
                               details={"available": sorted(_BUILDERS)})
    return _BUILDERS[name]()


def resolve_chain(requested: str = "auto") -> list[TranslationEngine]:
    """Ordered list of engines to try.

    When the remote endpoint is configured it goes FIRST on an "auto" chain.
    That is the whole point of enabling it: the local alternative is a 2.3 GB
    download this setup exists to avoid. It still falls back to whatever
    TRANSLATION_FALLBACK names, so a dead Colab tunnel degrades the job instead
    of failing it - at the cost of pulling the local model that one time.
    """
    if requested != "auto":
        names = [requested]
        if settings.translation_fallback not in ("none", requested):
            names.append(settings.translation_fallback)
    else:
        names = []
        if settings.remote_translation_enabled and settings.remote_url:
            names.append("remote")
        names.append(settings.translation_primary)
        if settings.translation_fallback not in ("none", settings.translation_primary):
            names.append(settings.translation_fallback)

    seen: set[str] = set()
    ordered = [n for n in names if not (n in seen or seen.add(n))]
    return [build(n) for n in ordered]


def available() -> list[str]:
    return sorted(_BUILDERS)
