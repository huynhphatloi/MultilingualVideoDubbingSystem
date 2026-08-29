"""Duration-aware translation helpers.

A literal translation is useless for dubbing if it takes 40% longer to read
than the original line. Two mechanisms are used:

1. **A priori budget** - before the first synthesis, estimate how many
   characters of the target language fit into the original slot.
2. **Empirical budget** - after synthesis we know the real spoken duration, so
   the budget is corrected by the measured ratio. This is far more accurate
   than any static table and is what ``adapt`` uses.
"""
from __future__ import annotations

import math

from app.core import languages

#: Average characters per syllable, by script family.
_CHARS_PER_SYLLABLE = {
    "cjk": 1.05,      # ja / zh / ko - one dense glyph per syllable
    "abugida": 2.4,   # hi / th / km / my ...
    "latin": 2.75,
}

_CJK = {"ja", "zh", "ko"}
_ABUGIDA = {"hi", "th", "km", "lo", "my", "bn", "ta", "te", "si", "ne", "ur"}


def _script_class(lang: str) -> str:
    code = languages.normalize(lang) or "en"
    if code in _CJK:
        return "cjk"
    if code in _ABUGIDA:
        return "abugida"
    return "latin"


def chars_per_second(lang: str) -> float:
    """Rough reading speed in characters per second for a language."""
    return languages.speech_rate(lang) * _CHARS_PER_SYLLABLE[_script_class(lang)]


#: How far the original speaker's pace may move the budget. Deliberately
#: narrow. It used to be 0.6..1.6, which was wrong in a specific and expensive
#: way: an actor delivering "Call it, Captain." in 0.86 s pushed the factor to
#: its 1.6 ceiling, so the translator was told 20 characters would fit. The
#: dub is not spoken by that actor - it is spoken by a TTS voice with a fixed
#: rate that needs ~1.2 s for those 20 characters. Every line came back ~60%
#: too long and the whole adapt/stretch machinery downstream spent its budget
#: fixing a number that was never achievable.
_SPEED_FACTOR_MIN = 0.85
_SPEED_FACTOR_MAX = 1.15


def estimate_budget(duration: float, target_language: str,
                    source_text: str | None = None,
                    source_language: str | None = None,
                    target_chars_per_second: float | None = None) -> int:
    """Characters of target text that fit into ``duration`` seconds - *when
    spoken by the TTS engine*, which is the only speaker that matters here.

    ``target_chars_per_second``, when supplied, is the rate measured from this
    job's own synthesis. It beats any table: it already accounts for the engine,
    the language and the voice. See ``measured_chars_per_second``.
    """
    rate = target_chars_per_second or chars_per_second(target_language)
    budget = duration * rate

    if source_text and source_language:
        source_rate = len(source_text) / max(duration, 0.2)
        table_source_rate = chars_per_second(source_language)
        if table_source_rate > 0:
            # A dense delivery does justify a slightly longer line - but only
            # slightly, because the TTS voice cannot match an actor's pace.
            budget *= _clamp(source_rate / table_source_rate,
                             _SPEED_FACTOR_MIN, _SPEED_FACTOR_MAX)

    return max(4, int(round(budget)))


def measured_chars_per_second(segments: list[dict]) -> float | None:
    """The TTS engine's real speaking rate, from this job's own takes.

    Static tables describe humans reading prose. MMS-TTS Vietnamese measured
    16.9 chars/s on the sample clip against a table value of 14.3, and other
    engine/language pairs are off in the other direction. Feeding the measured
    number back into the budget is what turns the adapt loop from a lottery
    into a correction.
    """
    chars = duration = 0.0
    for seg in segments:
        text = seg.get("translated_text") or ""
        generated = seg.get("generated_duration")
        if not text or not generated or generated <= 0:
            continue
        chars += len(text)
        duration += float(generated)
    if duration < 2.0:          # too little evidence to trust
        return None
    return chars / duration


def budget_for(duration: float, target_language: str,
               rate: float | None = None) -> int:
    """Plain slot -> characters, no source-side blending."""
    return max(4, int(round(duration * (rate or chars_per_second(target_language)))))


def budget_from_ratio(current_text: str, ratio: float,
                      target_ratio: float = 1.0) -> int:
    """Empirical correction.

    ``ratio = generated_duration / original_duration``. If the take was 24%
    too long, the next attempt gets ~24% fewer characters.
    """
    if ratio <= 0:
        return max(4, len(current_text))
    scale = _clamp(target_ratio / ratio, 0.45, 2.2)
    return max(4, int(round(len(current_text) * scale)))


def classify(ratio: float, accept_min: float, accept_max: float,
             stretch_min: float, stretch_max: float) -> str:
    """Map a duration ratio to the action the pipeline should take."""
    if accept_min <= ratio <= accept_max:
        return "accept"
    if stretch_min <= ratio <= stretch_max:
        return "stretch"
    return "retranslate"


def stretch_factor(ratio: float) -> float:
    """atempo factor that makes a take of ``ratio`` fit its slot exactly."""
    return _clamp(ratio, 0.5, 2.0)


def _clamp(value: float, low: float, high: float) -> float:
    if math.isnan(value):
        return low
    return max(low, min(high, value))
