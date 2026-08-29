"""Model router / fallback for speech generation.

    if language_supported_by_voice_clone and reference available:
        use voice cloning model
    else:
        use language-specific TTS
"""
from __future__ import annotations

import logging
from functools import lru_cache

from app.core import languages
from app.core.config import settings
from app.core.errors import UnsupportedLanguage
from app.services.tts.base import SpeechGenerator
from app.services.tts.f5 import F5Adapter
from app.services.tts.mms import MMSAdapter
from app.services.tts.remote import RemoteAdapter
from app.services.tts.xtts import XTTSAdapter

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _adapters() -> dict[str, SpeechGenerator]:
    """Every engine the router may pick, in preference order.

    Remote adapters go first so a configured GPU endpoint wins the chain;
    unconfigured they report supports() == False and cost nothing.

    REMOTE_TTS_ENGINES turns one endpoint into several named adapters
    (`remote:vixtts`, `remote:f5_vi`). They are ordered as listed, so the first
    id is the default preference and the rest are reachable by name through
    `force_model` - which is how a comparison run switches engines without
    touching the endpoint. An empty list keeps the single `remote` adapter and
    lets the server decide.
    """
    remotes = [RemoteAdapter(engine) for engine in settings.remote_tts_engine_list]
    if not remotes:
        remotes = [RemoteAdapter()]
    # Order is preference order for the automatic chain: cloning first (it is
    # the only way to keep the original actor), then MMS as the floor.
    local = (XTTSAdapter(), F5Adapter(), MMSAdapter())
    return {a.name: a for a in (*remotes, *local)}


def available() -> list[dict]:
    return [a.describe() for a in _adapters().values()]


def catalog(language: str, *, has_voice_reference: bool = True) -> list[dict]:
    """What the model picker should show for this language.

    Unavailable engines are RETURNED, not filtered out, each with the reason.
    That is deliberate: "XTTS-v2 - does not speak vi" teaches why a Vietnamese
    job sounds like one person, whereas a dropdown that silently contains only
    MMS makes the pipeline look like it has no other option. The UI greys them
    out; it does not hide them.

    `auto` leads the list. It is not an engine - it means "no force_model",
    which is the only mode with a fallback chain behind it. Every named choice
    runs alone by design, so a comparison cannot be contaminated by a silent
    fallback, and that trade is stated here rather than hidden.
    """
    entries: list[dict] = [{
        "id": "auto",
        "label": "Automatic — let the router choose",
        "blurb": ("Prefers a cloning engine when one speaks the language, and "
                  "falls back through the chain if it fails. The only mode "
                  "with a safety net."),
        "voice_cloning": None,
        "available": True,
        "reason": None,
    }]

    for adapter in _adapters().values():
        speaks = adapter.supports(language)
        needs_ref = adapter.supports_cloning and not has_voice_reference
        described = adapter.describe()
        reason = None
        if not speaks:
            # Each adapter explains its own "no" - see
            # SpeechGenerator.unavailable_reason.
            reason = adapter.unavailable_reason(language)
        elif needs_ref:
            reason = "needs a voice reference from the source video"

        entries.append({
            "id": adapter.name,
            "label": described.get("title") or adapter.name,
            "blurb": described.get("blurb") or "",
            "voice_cloning": adapter.supports_cloning,
            "available": speaks and not needs_ref,
            "reason": reason,
        })
    return entries


def resolve(language: str, *, has_voice_reference: bool,
            prefer_voice_clone: bool | None = None,
            force_model: str | None = None) -> list[SpeechGenerator]:
    """Ordered candidate list: first entry is the preferred engine."""
    adapters = _adapters()

    if force_model:
        if force_model not in adapters:
            raise UnsupportedLanguage(
                f"Unknown TTS model '{force_model}'",
                details={"available": sorted(adapters)},
            )
        return [adapters[force_model]]

    prefer = settings.tts_prefer_voice_clone if prefer_voice_clone is None else prefer_voice_clone
    cloning = [a for a in adapters.values() if a.supports_cloning and a.supports(language)]
    plain = [a for a in adapters.values() if not a.supports_cloning and a.supports(language)]

    chain: list[SpeechGenerator] = []
    if prefer and has_voice_reference:
        chain.extend(cloning)
    chain.extend(plain)
    if not prefer or not has_voice_reference:
        # Cloning models can still run last (they need a reference, so only if we have one).
        chain.extend(a for a in cloning if has_voice_reference and a not in chain)

    # De-duplicate while keeping order.
    seen: set[str] = set()
    ordered = [a for a in chain if not (a.name in seen or seen.add(a.name))]

    if not ordered:
        raise UnsupportedLanguage(
            f"No TTS engine can speak '{language}'.",
            details={
                "language": language,
                "voice_cloning_languages": sorted(languages.XTTS_LANGUAGES),
                "hint": "MMS-TTS covers most languages; check the ISO-639-3 checkpoint exists.",
            },
        )
    return ordered
