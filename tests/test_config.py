"""Job configuration: legacy parameters, compatibility rules, feature flags."""
from __future__ import annotations

import pytest

from core import config
from core.errors import InvalidRequest, ServiceError, UnsupportedLanguage


def build(**values):
    return config.build({"target_language": "vi", **values})


def test_defaults_match_the_pre_refactor_pipeline():
    built = build()
    assert built.asr.id == "small"
    assert built.translation.id == "nllb"
    assert built.tts.id == "mms"
    assert built.source_language is None
    features = built.features.public()
    assert features == {
        "diarization": False,
        "voice_cloning": False,
        "alignment": True,
        "source_separation": False,
    }


def test_legacy_parameter_names_still_select_models(all_installed):
    built = config.build({
        "target_language": "vi",
        "model": "large-v3-turbo",
        "translation_engine": "seamless",
        "tts_engine": "edge",
    })
    assert built.asr.id == "large-v3-turbo"
    assert built.translation.id == "seamless"
    assert built.tts.id == "edge"


def test_new_names_win_and_provider_plus_model_must_agree():
    built = config.build({
        "target_language": "vi",
        "asr_provider": "faster_whisper",
        "asr_model": "medium",
        "model": "tiny",
    })
    assert built.asr.id == "medium"
    with pytest.raises(InvalidRequest):
        config.build({
            "target_language": "vi", "asr_provider": "mms", "asr_model": "medium"
        })


def test_a_provider_alone_selects_its_first_model():
    assert config.build({"target_language": "vi", "tts_provider": "mms"}).tts.id == "mms"


def test_the_documented_invalid_combination_is_refused():
    """target_language=vi with tts_model=xtts_v2 must fail with a clear error."""
    with pytest.raises(UnsupportedLanguage) as failure:
        build(tts_model="xtts_v2")
    assert "does not support target language 'vi'" in str(failure.value)


def test_auto_detection_requires_a_recogniser_that_can_detect():
    with pytest.raises(InvalidRequest) as failure:
        build(asr_model="mms_asr")
    assert "cannot detect the spoken language" in str(failure.value)
    # Naming the language makes the same choice valid.
    assert build(asr_model="mms_asr", source_language="en").asr.id == "mms_asr"


def test_a_source_language_the_recogniser_cannot_read_is_refused():
    with pytest.raises(UnsupportedLanguage):
        build(model="tiny.en", source_language="vi")
    assert build(model="tiny.en", source_language="en").asr.id == "tiny.en"


def test_the_translation_model_must_cover_both_ends(all_installed):
    with pytest.raises(UnsupportedLanguage):
        config.build({
            "target_language": "si", "translation_model": "seamless",
            "source_language": "en", "tts_model": "edge",
        })


def test_translation_language_support_is_not_checked_when_it_is_skipped():
    """Source equal to target means no model runs, so nothing is validated."""
    built = config.build({
        "target_language": "en", "source_language": "en", "tts_model": "mms"
    })
    assert built.source_language == built.target_language


def test_voice_cloning_cannot_be_asked_of_an_engine_that_cannot_clone():
    with pytest.raises(InvalidRequest) as failure:
        build(tts_model="mms", enable_voice_cloning="true")
    assert "does not clone voices" in str(failure.value)


def test_voice_cloning_defaults_to_on_for_engines_that_need_a_reference(all_installed):
    built = config.build({"target_language": "vi", "tts_model": "vixtts"})
    assert built.features.voice_cloning is True
    assert built.public()["tts"]["voice_cloning"] is True


def test_cloning_cannot_be_turned_off_for_a_clone_only_engine(all_installed):
    with pytest.raises(InvalidRequest) as failure:
        build(tts_model="vixtts", enable_voice_cloning="false")
    assert "cannot be turned off" in str(failure.value)


def test_diarization_needs_an_installed_provider():
    with pytest.raises(ServiceError) as failure:
        build(enable_diarization="true")
    # Either the package is missing or the gated-weights token is: both are
    # reported, neither is silently ignored.
    assert failure.value.status_code == 503



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
    payload = build(model="large-v3-turbo").public()
    assert payload["asr"]["provider"] == "faster_whisper"
    assert payload["asr"]["model"] == "large-v3-turbo"
    assert payload["translation"]["model"] == "nllb"
    assert payload["tts"]["voice_cloning"] is False
    assert set(payload["features"]) == {
        "diarization", "voice_cloning", "alignment", "source_separation"
    }


def test_rebuild_restores_a_stored_job_with_its_detected_language():
    job = {"request": {"target_language": "vi"}, "source_language": "en"}
    assert config.rebuild(job).source_language == "en"


def test_confirm_source_language_rejects_a_detection_the_models_cannot_use():
    built = config.build({"target_language": "en", "model": "large-v3", "tts_model": "mms"})
    built.confirm_source_language("vi")  # supported by Whisper and NLLB
    narrow = config.build({
        "target_language": "en", "model": "large-v3", "tts_model": "mms",
        "translation_model": "seamless",
    })
    with pytest.raises(UnsupportedLanguage):
        narrow.confirm_source_language("si")
