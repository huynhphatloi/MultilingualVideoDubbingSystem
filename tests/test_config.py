from __future__ import annotations

import pytest

from core import config
from core.errors import InvalidRequest, ServiceError, UnsupportedLanguage


def build(**values):
    return config.build({"target_language": "vi", **values})


def test_defaults_select_the_base_models():
    built = build()
    assert built.asr.id == "small"
    assert built.translation.id == "nllb"
    assert built.tts.id == "mms"
    assert built.source_language is None
    features = built.features.public()
    assert features == {
        "diarization": False,
        "alignment": True,
        "source_separation": False,
        "multi_voice": False,
    }


def test_provider_and_model_must_agree():
    built = config.build({
        "target_language": "vi",
        "asr_provider": "faster_whisper",
        "asr_model": "medium",
    })
    assert built.asr.id == "medium"
    with pytest.raises(InvalidRequest):
        config.build({
            "target_language": "vi", "asr_provider": "mms", "asr_model": "medium"
        })


def test_a_provider_alone_selects_its_first_model():
    assert config.build({"target_language": "vi", "tts_provider": "mms"}).tts.id == "mms"


def test_the_documented_invalid_combination_is_refused():
    with pytest.raises(UnsupportedLanguage) as failure:
        build(tts_model="mms", target_language="ja")
    assert "does not support target language 'ja'" in str(failure.value)


def test_auto_detection_requires_a_recogniser_that_can_detect():
    with pytest.raises(InvalidRequest) as failure:
        build(asr_model="mms_asr")
    assert "cannot detect the spoken language" in str(failure.value)
    assert build(asr_model="mms_asr", source_language="en").asr.id == "mms_asr"


def test_a_source_language_the_recogniser_cannot_read_is_refused():
    with pytest.raises(UnsupportedLanguage):
        build(asr_model="tiny.en", source_language="vi")
    assert build(asr_model="tiny.en", source_language="en").asr.id == "tiny.en"


def test_the_translation_model_must_cover_both_ends(all_installed):
    with pytest.raises(UnsupportedLanguage):
        config.build({
            "target_language": "si", "translation_model": "seamless",
            "source_language": "en", "tts_model": "edge",
        })


def test_translation_language_support_is_not_checked_when_it_is_skipped():
    built = config.build({
        "target_language": "en", "source_language": "en", "tts_model": "mms"
    })
    assert built.source_language == built.target_language


def test_diarization_needs_an_installed_provider():
    with pytest.raises(ServiceError) as failure:
        build(enable_diarization="true")
    assert failure.value.status_code == 503


def test_multi_voice_flag_selects_edge_and_requires_diarization(
    monkeypatch, all_installed
):
    monkeypatch.setenv("DUBFLOW_MULTI_VOICE", "true")
    built = build(enable_diarization="false")

    assert built.tts.id == "edge"
    assert built.features.multi_voice is True
    assert built.features.diarization is True
    assert built.diarization.id == "pyannote_3_1"


def test_multi_voice_flag_rejects_a_single_voice_model(monkeypatch, all_installed):
    monkeypatch.setenv("DUBFLOW_MULTI_VOICE", "true")
    with pytest.raises(InvalidRequest) as failure:
        build(tts_model="mms")
    assert "requires a TTS model with multi-speaker support" in str(failure.value)


def test_alignment_limits_are_validated_and_carried():
    built = build(min_speed="0.8", max_speed="1.5")
    assert built.limits.min_speed == 0.8
    assert built.limits.max_speed == 1.5
    assert built.public()["alignment_limits"]["max_speed"] == 1.5
    with pytest.raises(InvalidRequest):
        build(max_speed="9")
    with pytest.raises(InvalidRequest):
        build(min_speed="0.1")


def test_unknown_languages_are_refused():
    with pytest.raises(InvalidRequest):
        config.build({"target_language": "klingon"})


def test_public_shape_matches_the_documented_job_config():
    payload = build(asr_model="large-v3-turbo").public()
    assert payload["asr"]["provider"] == "faster_whisper"
    assert payload["asr"]["model"] == "large-v3-turbo"
    assert payload["translation"]["model"] == "nllb"
    assert set(payload["features"]) == {
        "diarization", "alignment", "source_separation", "multi_voice"
    }
