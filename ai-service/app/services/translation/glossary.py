"""Terms the translator must not reinterpret.

NLLB is already good at proper nouns - measured on a 16-name battery it kept
15 of 16 (David Malan, CS50, Harvard, edX, YouTube, Apple TV, Scratch, Barton,
Stark, Legolas, Thor, Hulk, Fury, Romanoff, Rogers all survived untouched). So
this module deliberately does NOT try to protect every capitalised word.

Blanket protection actively makes things worse. Swapping every name for a
marker and restoring it afterwards was measured against the same sentences:

    raw:         "Được rồi, hãy cất cánh đi, Legolas."
    protected:   "Được rồi, tốt hơn là clench lên, @@0@@."   <- "clench" untranslated
    raw:         "Tên tôi là David Malan, và đây là CS50, Đại học Harvard."
    protected:   "Tên tôi là @@0@@, và đây là @@1@@, @@2@@ của trường đại học"

The marker starves the model of context and the surrounding grammar degrades.

What actually fails is a narrow class: a word that is BOTH an ordinary noun and
a name in this material. "Captain" is the canonical case - as a rank it really
is "Đại úy", but when it is what people call Steve Rogers it has to stay
"Captain". No amount of model quality fixes that, because the model cannot know
which one this film means. Only the operator knows, so only the operator's
list is protected.

Format - ``config/glossary.json``::

    {
      "en": {
        "Captain": "Captain",          // keep as-is
        "Cap": "Đội trưởng",           // force a specific rendering
        "Scratch": "Scratch"
      },
      "*": { "CS50": "CS50" }          // applies to every source language
    }
"""
from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path

log = logging.getLogger(__name__)

#: Marker shape used for the round trip. Chosen by measurement: of the shapes
#: tried (``@@N@@``, ``{N}``, ``XNX``, ``NNN``, ``[N]``), this one and ``[N]``
#: and ``XNX`` came back intact on every sentence; ``#N#`` and ``%N%`` did not.
#: NLLB does sometimes emit a stray extra delimiter (``@@1@@@``), which is why
#: the restore pattern below is deliberately loose about them.
_MARKER = "@@{}@@"
_MARKER_RE = re.compile(r"@+\s*(\d+)\s*@+")

DEFAULT_PATH = Path(__file__).resolve().parents[4] / "config" / "glossary.json"


@lru_cache(maxsize=8)
def load(path: str | None = None) -> dict[str, dict[str, str]]:
    """Read the glossary file. A missing file just means 'no protected terms'."""
    target = Path(path) if path else DEFAULT_PATH
    if not target.exists():
        return {}
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("glossary at %s is unreadable (%s) - continuing without it",
                    target, exc)
        return {}
    return {
        str(lang): {str(k): str(v) for k, v in (terms or {}).items()}
        for lang, terms in raw.items()
        if isinstance(terms, dict)
    }


def terms_for(source_language: str, path: str | None = None) -> dict[str, str]:
    """Protected terms for one source language, merged with the '*' entries."""
    table = load(path)
    merged = dict(table.get("*", {}))
    merged.update(table.get(source_language, {}))
    return merged


def protect(text: str, terms: dict[str, str]) -> tuple[str, dict[str, str]]:
    """Replace protected terms with markers. Returns (masked text, marker map).

    Longest term first, so "Apple TV" is matched before "Apple". Matching is
    whole-word and case-sensitive: lowercase "stark" in "a stark contrast" is a
    different word from the character "Stark".
    """
    if not terms or not text:
        return text, {}

    mapping: dict[str, str] = {}
    masked = text
    for term in sorted(terms, key=len, reverse=True):
        if term not in masked:
            continue
        pattern = re.compile(rf"(?<!\w){re.escape(term)}(?!\w)")
        if not pattern.search(masked):
            continue
        marker = _MARKER.format(len(mapping))
        mapping[marker] = terms[term]
        masked = pattern.sub(marker, masked)
    return masked, mapping


def restore(text: str, mapping: dict[str, str]) -> str:
    """Put the protected terms back, tolerating the model's stray delimiters."""
    if not mapping or not text:
        return text

    by_index = {}
    for marker, replacement in mapping.items():
        match = _MARKER_RE.fullmatch(marker)
        if match:
            by_index[match.group(1)] = replacement

    def _swap(match: re.Match) -> str:
        return by_index.get(match.group(1), match.group(0))

    restored = _MARKER_RE.sub(_swap, text)

    # A marker the model dropped entirely means the name is gone from the line.
    # Do NOT bolt it back on: this text is about to be SPOKEN, and
    # "Và Hulk Captain Smash" is a worse thing to hear than a missing name.
    # Report it instead - a term that keeps vanishing is a sign it should be
    # rendered rather than preserved.
    missing = [v for k, v in by_index.items()
               if v and v not in restored and _MARKER.format(k) not in text]
    if missing:
        log.warning("glossary term(s) %s did not survive translation of %r",
                    missing, text[:80])
    return _recase(restored)


def _recase(text: str) -> str:
    """Re-capitalise the opening word.

    A masked sentence often comes back starting lowercase, because the model saw
    a marker rather than a capitalised word in that position - "gọi nó,
    Captain." instead of "Gọi nó, Captain.". Harmless in a subtitle, but this
    text also seeds the TTS engine, and it looks like a bug to anyone reading
    the .srt.
    """
    stripped = text.lstrip()
    if not stripped or not stripped[0].isalpha() or stripped[0].isupper():
        return text
    lead = len(text) - len(stripped)
    return text[:lead] + stripped[0].upper() + stripped[1:]


