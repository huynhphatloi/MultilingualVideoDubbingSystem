from __future__ import annotations

import io
import json

import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")


@pytest.fixture(scope="module")
def client():  # noqa: ANN201
    import server

    with fastapi_testclient.TestClient(server.app) as running:
        yield running


def test_capabilities_is_the_source_of_truth(client):
    payload = client.get("/capabilities").json()
    assert payload["tasks"] == [
        "asr", "translation", "tts", "diarization", "separation"
    ]
    assert payload["stages"][0] == "extract" and payload["stages"][-1] == "render"
    assert payload["defaults"]["asr"] == "small"
    assert payload["feature_flags"]["multi_voice"] is False
    asr = {
        model["id"]: model
        for group in payload["providers"]["asr"] for model in group["models"]
    }
    assert asr["large-v3"]["available"] is True
    assert asr["large-v3"]["supports_timestamps"] is True
    assert asr["tiny.en"]["languages"] == ["en"]


def test_health_keeps_the_keys_the_previous_version_published(client):
    payload = client.get("/health").json()
    assert payload["status"] == "ready"
    assert set(payload["whisper_models"]) >= {"small", "large-v3", "large-v3-turbo"}
    assert set(payload["translation_engines"]) >= {"nllb", "seamless"}
    engines = payload["tts_engines"]
    assert {"mms", "edge"} <= set(engines)
    assert engines["mms"]["available"] is True
    assert payload["feature_flags"]["multi_voice"] is False
    assert payload["languages"][0]["code"]


def test_languages_endpoint(client):
    codes = {row["code"] for row in client.get("/languages").json()["languages"]}
    assert {"en", "vi", "ja"} <= codes


def test_validate_accepts_the_old_parameter_names(client):
    response = client.post("/validate", json={
        "target_language": "vi", "model": "large-v3", "translation_engine": "nllb",
        "tts_engine": "mms",
    })
    assert response.status_code == 200
    config = response.json()["config"]
    assert config["asr"]["model"] == "large-v3"
    assert config["features"]["alignment"] is True


def test_validate_refuses_an_impossible_pairing_with_a_readable_message(client):
    response = client.post("/validate", json={
        "target_language": "ja", "tts_model": "mms"
    })
    assert response.status_code == 400
    assert "does not support target language 'ja'" in response.json()["detail"]


def test_translate_skips_when_the_languages_match(client):
    response = client.post("/translate", json={
        "source_language": "vi", "target_language": "vi", "texts": ["xin chao"]
    })
    payload = response.json()
    assert payload["translations"] == ["xin chao"]
    assert payload["skipped"] is True
    assert payload["translation_engine"] == "nllb"


def test_translate_rejects_an_empty_batch(client):
    response = client.post("/translate", json={
        "source_language": "en", "target_language": "vi", "texts": []
    })
    assert response.status_code == 400


def test_align_plans_without_touching_audio(client):
    response = client.post("/align", json={
        "segments": [
            {"id": 0, "start": 0.0, "end": 3.5, "tts_duration_raw": 4.4},
            {"id": 1, "start": 4.0, "end": 6.0, "tts_duration_raw": 1.0},
        ],
        "media_duration": 10.0,
    })
    plans = response.json()["plans"]
    assert plans[0]["status"] == "aligned"
    assert plans[0]["speed"] > 1.0
    assert plans[1]["status"] == "fits"


def test_transcribe_rejects_auto_for_a_recogniser_that_cannot_detect(client):
    response = client.post(
        "/transcribe",
        files={"audio": ("a.wav", io.BytesIO(b"RIFF"), "audio/wav")},
        data={"language": "auto", "model": "mms_asr"},
    )
    assert response.status_code == 400
    assert "cannot detect the spoken language" in response.json()["detail"]


def test_transcribe_rejects_malformed_turns(client):
    response = client.post(
        "/transcribe",
        files={"audio": ("a.wav", io.BytesIO(b"RIFF"), "audio/wav")},
        data={"language": "en", "model": "small", "turns": json.dumps([{"start": 0}])},
    )
    assert response.status_code == 400


def test_unknown_job_ids(client):
    assert client.get("/jobs/aaaaaaaaaaaa").status_code == 404
    assert client.get("/jobs/short").status_code == 400


def test_synthesize_refuses_a_language_the_engine_cannot_speak(client):
    response = client.post("/synthesize", data={
        "text": "hello", "language": "ja", "engine": "mms"
    })
    assert response.status_code == 400
    assert "does not support target language 'ja'" in response.json()["detail"]


def test_synthesize_cannot_enable_multi_voice_when_the_startup_flag_is_off(client):
    response = client.post("/synthesize", data={
        "text": "hello", "language": "vi", "multi_voice": "true"
    })
    assert response.status_code == 400
    assert "Start the AI service with DUBFLOW_MULTI_VOICE=true" in response.json()["detail"]


def test_authentication_is_still_enforced_when_a_token_is_set(client, monkeypatch):
    import server

    monkeypatch.setattr(server, "AUTH_TOKEN", "secret")
    assert client.get("/jobs").status_code == 401
    assert client.get(
        "/jobs", headers={"Authorization": "Bearer secret"}
    ).status_code == 200
    assert client.get("/health").status_code == 200


def test_creating_a_job_reads_both_the_old_and_new_form_fields(client, monkeypatch, tmp_path):
    import jobs

    monkeypatch.setattr(jobs, "ROOT", tmp_path)
    monkeypatch.setattr(jobs, "enqueue", lambda job_id: 0)

    response = client.post(
        "/jobs",
        files={"video": ("clip.mp4", b"not really a video", "video/mp4")},
        data={
            "target_language": "vi",
            "source_language": "en",
            "model": "large-v3-turbo",
            "translation_engine": "nllb",
            "tts_engine": "mms",
            "enable_alignment": "false",
        },
    )
    assert response.status_code == 202, response.text
    payload = response.json()
    assert payload["status"] == "queued"
    assert payload["whisper_model"] == "large-v3-turbo"
    assert payload["config"]["features"]["alignment"] is False
    assert payload["progress"] == "0/7"
    assert client.get(f"/jobs/{payload['job_id']}").status_code == 200


def test_creating_a_job_refuses_an_unsupported_container(client, monkeypatch, tmp_path):
    import jobs

    monkeypatch.setattr(jobs, "ROOT", tmp_path)
    response = client.post(
        "/jobs",
        files={"video": ("clip.avi", b"x", "video/x-msvideo")},
        data={"target_language": "vi"},
    )
    assert response.status_code == 400
    assert "Unsupported video container" in response.json()["detail"]
