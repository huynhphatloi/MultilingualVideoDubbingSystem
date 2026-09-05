"""Registry invariants, availability reporting and language filtering."""
from __future__ import annotations

import pytest

from core.errors import InvalidRequest, MissingDependency, UnsupportedLanguage
from dubflow_core import languages as L


def test_no_internal_inconsistencies(registry):
    assert registry.consistency_problems() == []


def test_every_task_has_a_registry(registry):
    assert set(registry.REGISTRIES) == {
        "asr", "translation", "tts", "diarization", "separation", "lipsync"
    }


def test_no_model_claims_a_language_the_application_cannot_name(registry):
    for spec in registry.every_model():
        assert set(spec.languages) <= set(L.CODES), spec.id


def test_the_old_engine_names_are_still_registered(registry):
    """The pre-refactor API accepted these; they have to keep resolving."""
    for checkpoint in (
        "tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium",
        "medium.en", "large-v2", "large-v3", "large-v3-turbo",
        "distil-small.en", "distil-medium.en", "distil-large-v3",
    ):
        assert registry.registry("asr").has(checkpoint)
    for engine in ("nllb", "seamless"):
        assert registry.registry("translation").has(engine)
    for voice in ("mms", "edge", "piper", "xtts_v2", "vixtts", "f5_vi", "f5_base"):
        assert registry.registry("tts").has(voice)


def test_english_only_checkpoints_are_marked_as_such(registry):
    asr = registry.registry("asr")
    for checkpoint in ("tiny.en", "distil-large-v3", "parakeet_tdt_0.6b_v2"):
        spec = asr.get(checkpoint)
        assert spec.languages == ("en",), checkpoint
        assert not spec.supports_language_detection
    assert asr.get("large-v3").languages == L.ALL


def test_only_models_that_report_the_language_claim_detection(registry):
    """A model that transcribes without being told the language is not a
    detector unless it also says which language it heard - the translation
    stage needs that answer."""
    asr = registry.registry("asr")
    assert asr.get("large-v3").supports_language_detection
    assert asr.get("sensevoice_small").supports_language_detection
    for silent in ("mms_asr", "seamless_asr", "parakeet_tdt_0.6b_v3"):
        assert not asr.get(silent).supports_language_detection, silent


def test_models_without_timestamps_are_flagged(registry):
    asr = registry.registry("asr")
    assert asr.get("large-v3").supports_timestamps
    for windowed in ("seamless_asr", "mms_asr", "sensevoice_small"):
        assert not asr.get(windowed).supports_timestamps


def test_documented_language_gaps(registry):
    """Coverage that was verified against each model's own card."""
    tts = registry.registry("tts")
    assert "vi" not in tts.get("xtts_v2").languages
    assert "vi" not in tts.get("chatterbox").languages
    assert tts.get("vixtts").languages == ("vi",)
    assert set(tts.get("f5_base").languages) == {"en", "zh"}
    assert set(tts.get("kokoro").languages) == {"en", "es", "fr", "hi", "it", "ja", "pt", "zh"}
    # MMS-TTS publishes no checkpoint for these, so the pipeline must not offer
    # them; before the registry existed the job failed with a 404 from the Hub.
    for missing in ("ja", "zh", "it", "cs", "da", "no"):
        assert missing not in tts.get("mms").languages
    # SeamlessM4T's speech input covers everything but Malay and Sinhala.
    assert set(registry.registry("asr").get("seamless_asr").languages) == set(
        L.exclude("ms", "si")
    )


def test_cloning_engines_declare_their_reference_need(registry):
    tts = registry.registry("tts")
    for cloner in ("xtts_v2", "vixtts", "f5_vi", "f5_base", "cosyvoice2", "vieneu"):
        spec = tts.get(cloner)
        assert spec.supports_voice_cloning and spec.reference_required, cloner
    for stock in ("mms", "edge", "piper", "kokoro"):
        assert not tts.get(stock).supports_voice_cloning, stock
    # Chatterbox clones but also has a built-in voice, so a reference is optional.
    chatterbox = tts.get("chatterbox")
    assert chatterbox.supports_voice_cloning and not chatterbox.reference_required


def test_licences_are_recorded_for_the_restricted_models(registry):
    assert registry.registry("asr").get("mms_asr").license == "cc-by-nc-4.0"
    assert registry.registry("translation").get("nllb").license == "cc-by-nc-4.0"
    assert registry.registry("tts").get("kokoro").license == "apache-2.0"
    assert registry.registry("tts").get("chatterbox").license == "mit"


def test_unavailable_models_report_why_instead_of_disappearing(registry):
    spec = registry.registry("tts").get("kokoro")
    if not spec.available():
        assert "kokoro" in (spec.unavailable_reason() or "")
        assert spec.public()["available"] is False
        assert spec.public()["installed"] is False
    with pytest.raises(MissingDependency):
        registry.registry("tts").require_available(spec)


def test_capabilities_reports_every_task(registry):
    payload = registry.capabilities()
    assert set(payload["providers"]) == set(registry.REGISTRIES)
    assert payload["languages"][0]["code"] in L.LANGUAGES
    for groups in payload["providers"].values():
        for group in groups:
            for model in group["models"]:
                assert {"id", "available", "installed", "languages", "license"} <= set(model)


def test_resolve_accepts_a_provider_a_model_or_both(registry):
    asr = registry.registry("asr")
    assert asr.resolve(model_id="large-v3").id == "large-v3"
    assert asr.resolve(provider_id="faster_whisper").provider == "faster_whisper"
    with pytest.raises(InvalidRequest):
        asr.resolve(provider_id="mms", model_id="large-v3")
    with pytest.raises(InvalidRequest):
        asr.resolve(model_id="does-not-exist")


def test_language_rejection_names_a_working_alternative(registry):
    tts = registry.registry("tts")
    with pytest.raises(UnsupportedLanguage) as failure:
        tts.require_language(tts.get("xtts_v2"), "vi", "target")
    message = str(failure.value)
    assert "does not support target language 'vi'" in message
    assert "en, es, fr" in message


def test_lipsync_registers_nothing_and_says_so(registry):
    assert registry.registry("lipsync").ids() == []
    assert registry.capabilities()["providers"]["lipsync"] == []
