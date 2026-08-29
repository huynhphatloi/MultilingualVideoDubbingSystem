"""A synthesis stage that produces nothing must SAY so.

Returning 200 with `synthesized: 0` is how a Japanese job painted the progress
UI green for 227 seconds of "Speech generation", then failed two stages later
with "No generated audio to synchronise - run synthesis first." - a true
sentence that names the wrong stage and hides the real cause (coqui-tts was
installed without its Japanese tokenizer, and MMS-TTS has no jpn checkpoint).
"""
from __future__ import annotations

import pytest
from app.core.errors import SynthesisError
from app.services.tts.service import (
    _FAIL_FAST_AFTER,
    _describe,
    _distinct,
    _language_hint,
)


# ------------------------------------------------------------- diagnostics --
def test_distinct_collapses_identical_failures():
    failures = [{"segment_id": i, "error": "boom"} for i in range(32)]
    assert _distinct(failures) == ["32x boom"]


def test_distinct_keeps_each_separate_cause():
    failures = [
        {"segment_id": 0, "error": "no mms checkpoint"},
        {"segment_id": 1, "error": "no cutlet"},
        {"segment_id": 2, "error": "no cutlet"},
    ]
    assert sorted(_distinct(failures)) == ["1x no mms checkpoint", "2x no cutlet"]


def test_describe_flattens_every_engine_attempt():
    """The chain is [xtts, mms]; showing only the last error blames MMS."""
    exc = SynthesisError("All TTS engines failed for segment 0",
                         details={"attempts": ["xtts_v2: No module named 'cutlet'",
                                               "mms_tts: no checkpoint for jpn"]})
    described = _describe(exc)
    assert "cutlet" in described
    assert "mms_tts" in described


def test_describe_passes_through_a_plain_error():
    assert _describe(ValueError("plain")) == "plain"


# ------------------------------------------------------------------- hints --
@pytest.mark.parametrize(
    ("language", "needle"),
    [("ja", "coqui-tts[ja]"), ("zh", "coqui-tts[zh]"), ("ko", "coqui-tts[ko]")],
)
def test_languages_needing_extra_packages_name_the_install(language, needle):
    assert needle in _language_hint(language)


def test_japanese_hint_says_there_is_no_mms_fallback():
    hint = _language_hint("ja")
    assert "MMS-TTS has no 'jpn' checkpoint" in hint


def test_unknown_language_still_gets_actionable_advice():
    hint = _language_hint("xx")
    assert "XTTS-v2" in hint and "mms-tts-" in hint


# --------------------------------------------------------------- fail fast --
def test_fail_fast_threshold_is_small_but_not_one():
    # 1 would abort on a single unlucky line; the point is to catch a
    # *configuration* fault, which fails every line identically.
    assert 2 <= _FAIL_FAST_AFTER <= 5


# ------------------------------------------------------------- job manifest --
def test_tally_counts_engines_across_the_whole_job():
    """The manifest must survive the adapt loop's final no-op synthesize call."""
    from app.services.tts.service import _tally

    segments = [
        {"tts_model": "xtts_v2", "generated_audio": "k0"},
        {"tts_model": "xtts_v2", "generated_audio": "k1"},
        {"tts_model": "mms_tts", "generated_audio": "k2"},
        {"tts_model": "xtts_v2", "generated_audio": None},   # failed -> not counted
        {"tts_model": None, "generated_audio": None},
    ]
    assert _tally(segments, "tts_model") == {"xtts_v2": 2, "mms_tts": 1}


def test_tally_of_an_empty_job_is_empty():
    from app.services.tts.service import _tally

    assert _tally([], "tts_model") == {}
