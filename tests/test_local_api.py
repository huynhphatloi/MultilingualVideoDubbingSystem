from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ai-service"))

ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="FFmpeg is required for these stages"
)

SEGMENTS = [
    {"id": 0, "speaker_id": "SPEAKER_00", "start": 0.5, "end": 3.0, "duration": 2.5,
     "source_text": "Good morning everyone", "translated_text": None, "tts_file": None},
    {"id": 1, "speaker_id": "SPEAKER_00", "start": 3.5, "end": 6.5, "duration": 3.0,
     "source_text": "Thank you for having me", "translated_text": None, "tts_file": None},
]
TURNS = [
    {"speaker_id": "SPEAKER_00", "start": 0.4, "end": 3.1},
    {"speaker_id": "SPEAKER_01", "start": 3.4, "end": 6.6},
]
CONFIG = {
    "source_language": "auto",
    "target_language": "vi",
    "asr": {"provider": "faster_whisper", "model": "small"},
    "translation": {"provider": "nllb", "model": "nllb"},
    "tts": {"provider": "edge", "model": "edge"},
    "features": {"diarization": True, "alignment": True,
                 "source_separation": False},
    "diarization": {"provider": "pyannote", "model": "pyannote_3_1"},
    "alignment_limits": {"min_speed": 0.75, "max_speed": 1.35, "tolerance": 0.05,
                         "allow_stretch": False},
}


class FakeResponse:
    def __init__(self, payload=None, content=b"", headers=None):  # noqa: ANN001
        self._payload = payload
        self.content = content
        self.headers = headers or {}
        self.status_code = 200

    def json(self):  # noqa: ANN201
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def make_video(path: Path, seconds: float = 8.0) -> Path:
    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"testsrc=duration={seconds}:size=320x240:rate=10",
        "-f", "lavfi", "-i", f"sine=frequency=300:duration={seconds}",
        "-c:v", "mpeg4", "-c:a", "aac", "-shortest", str(path),
    ], check=True, capture_output=True)
    return path


def tone_bytes(seconds: float) -> bytes:
    result = subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds:.3f}",
        "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", "-f", "wav", "pipe:1",
    ], check=True, capture_output=True).stdout
    return result


@pytest.fixture
def service(tmp_path, monkeypatch):  # noqa: ANN201
    import app

    monkeypatch.setattr(app, "ROOT", tmp_path)
    calls = []
    speech = {"Good morning everyone": 2.0, "Thank you for having me": 5.4}

    def fake_request(path, method="POST", **kwargs):  # noqa: ANN001, ANN202
        calls.append((path, kwargs.get("data") or kwargs.get("json") or {}))
        if path == "/validate":
            return FakeResponse({"valid": True, "config": json.loads(json.dumps(CONFIG))})
        if path == "/capabilities":
            return FakeResponse({"providers": {}, "languages": [], "device": "cpu"})
        if path == "/diarize":
            return FakeResponse({"turns": TURNS, "speakers": ["SPEAKER_00", "SPEAKER_01"],
                                 "diarization_model": "test/diarization"})
        if path == "/transcribe":
            return FakeResponse({
                "source_language": "en",
                "segments": json.loads(json.dumps(SEGMENTS)),
                "asr_model": "test/asr",
            })
        if path == "/translate":
            texts = kwargs["json"]["texts"]
            return FakeResponse({"translations": [f"[vi] {text}" for text in texts]})
        if path == "/synthesize":
            source = kwargs["data"]["text"].split("] ", 1)[-1]
            headers = {"x-tts-model": "test/tts"}
            speaker = kwargs["data"].get("speaker_id")
            if speaker:
                headers["x-tts-voice"] = f"voice-{speaker}"
            return FakeResponse(
                content=tone_bytes(speech.get(source, 1.0)),
                headers=headers,
            )
        raise AssertionError(f"unexpected backend call {path}")

    monkeypatch.setattr(app, "_colab_request", fake_request)
    monkeypatch.setattr(app, "_resolve_backend", lambda force=False: ("test", "http://x", ""))
    client = fastapi_testclient.TestClient(app.app)
    return client, calls, tmp_path


def upload(client, tmp_path, **fields):  # noqa: ANN001, ANN201
    video = make_video(tmp_path / "source.mp4")
    with video.open("rb") as handle:
        response = client.post(
            "/jobs/upload",
            files={"file": ("source.mp4", handle, "video/mp4")},
            data={"target_language": "vi", **fields},
        )
    assert response.status_code == 200, response.text
    return response.json()["job_id"]


def stage(client, name, job_id):  # noqa: ANN001, ANN201
    response = client.post(f"/stages/{name}", json={"job_id": job_id})
    assert response.status_code == 200, f"{name}: {response.text}"
    return response.json()


@ffmpeg
def test_the_whole_n8n_route_runs(service):
    client, _, tmp_path = service
    job_id = upload(client, tmp_path)

    results = {name: stage(client, name, job_id) for name in [
        "extract", "diarize", "transcribe", "merge_segments", "translate",
        "synthesize", "align", "separate", "mix", "render",
    ]}

    assert results["diarize"]["status"] == "completed"
    assert results["separate"]["status"] == "skipped"
    assert results["mix"]["mode"] == "voice-over"

    job = client.get(f"/jobs/{job_id}").json()
    assert job["status"] == "completed"
    assert job["completed_stages"] == [
        "upload", "extract", "diarize", "transcribe", "merge_segments", "translate",
        "synthesize", "align", "mix", "render",
    ]
    output = tmp_path / job_id / job["files"]["output"]
    assert output.exists() and output.stat().st_size > 0
    assert client.get(f"/jobs/{job_id}/download").status_code == 200
    assert client.get(f"/jobs/{job_id}/subtitle").status_code == 200


@ffmpeg
def test_alignment_retimes_the_clip_that_overruns(service):
    client, _, tmp_path = service
    job_id = upload(client, tmp_path)
    for name in ("extract", "transcribe", "merge_segments", "translate", "synthesize"):
        stage(client, name, job_id)
    before = client.get(f"/jobs/{job_id}").json()["segments"]
    assert before[1]["tts_duration_raw"] == pytest.approx(5.4, abs=0.2)

    stage(client, "align", job_id)
    after = client.get(f"/jobs/{job_id}").json()["segments"]
    assert after[0]["alignment_status"] == "fits"
    assert after[1]["alignment_status"] in {"aligned", "clamped"}
    assert after[1]["alignment_speed"] > 1.0
    assert after[1]["tts_duration_final"] < after[1]["tts_duration_raw"]


@ffmpeg
def test_diarization_labels_every_segment_with_its_speaker(service, monkeypatch):
    """Diarization labels are carried into the per-speaker voice stage."""
    client, calls, tmp_path = service
    import app

    diarized = json.loads(json.dumps(CONFIG))
    diarized["features"]["diarization"] = True
    diarized["diarization"] = {"provider": "pyannote", "model": "pyannote_3_1"}
    original = app._colab_request

    def routed(path, method="POST", **kwargs):  # noqa: ANN001, ANN202
        if path == "/validate":
            return FakeResponse({"valid": True, "config": diarized})
        return original(path, method, **kwargs)

    monkeypatch.setattr(app, "_colab_request", routed)

    job_id = upload(client, tmp_path, enable_diarization="true")
    for name in ("extract", "diarize", "transcribe", "merge_segments", "translate",
                 "synthesize"):
        stage(client, name, job_id)

    job = client.get(f"/jobs/{job_id}").json()
    assert job["speakers"] == ["SPEAKER_00", "SPEAKER_01"]
    assert [segment["speaker_id"] for segment in job["segments"]] == [
        "SPEAKER_00", "SPEAKER_01"
    ]
    assert job["speaker_summary"], "each speaker's share is recorded"
    # Every line was still voiced, one clip per segment.
    assert all(segment["tts_file"] for segment in job["segments"])
    assert len([1 for path, _ in calls if path == "/synthesize"]) == len(job["segments"])


@ffmpeg
def test_synthesis_sends_speaker_ids_and_records_the_voice_map(service, monkeypatch):
    client, calls, tmp_path = service
    import app

    configured = json.loads(json.dumps(CONFIG))
    configured["features"]["diarization"] = True
    configured["diarization"] = {"provider": "pyannote", "model": "pyannote_3_1"}
    configured["tts"] = {"provider": "edge", "model": "edge"}
    original = app._colab_request

    def routed(path, method="POST", **kwargs):  # noqa: ANN001, ANN202
        if path == "/validate":
            return FakeResponse({"valid": True, "config": configured})
        return original(path, method, **kwargs)

    monkeypatch.setattr(app, "_colab_request", routed)
    job_id = upload(client, tmp_path)
    for name in (
        "extract", "diarize", "transcribe", "merge_segments", "translate", "synthesize"
    ):
        stage(client, name, job_id)

    job = client.get(f"/jobs/{job_id}").json()
    voice_calls = [data for path, data in calls if path == "/synthesize"]
    assert [call["speaker_id"] for call in voice_calls] == ["SPEAKER_00", "SPEAKER_01"]
    assert all(set(call) == {
        "text", "language", "speed", "provider", "model", "speaker_id"
    } for call in voice_calls)
    assert job["speaker_voice_map"] == {
        "SPEAKER_00": "voice-SPEAKER_00",
        "SPEAKER_01": "voice-SPEAKER_01",
    }
    assert [segment["tts_voice"] for segment in job["segments"]] == [
        "voice-SPEAKER_00", "voice-SPEAKER_01"
    ]


@ffmpeg
def test_upload_refuses_a_configuration_the_backend_rejects(service, monkeypatch):
    client, _, tmp_path = service
    import app
    from fastapi import HTTPException

    def refuse(path, method="POST", **kwargs):  # noqa: ANN001, ANN202
        raise HTTPException(502, "AI backend 'test' returned 400: 'XTTS v2' does not "
                                 "support target language 'vi'.")

    monkeypatch.setattr(app, "_colab_request", refuse)
    video = make_video(tmp_path / "bad.mp4")
    with video.open("rb") as handle:
        response = client.post(
            "/jobs/upload",
            files={"file": ("bad.mp4", handle, "video/mp4")},
            data={"target_language": "ja", "tts_model": "mms"},
        )
    assert response.status_code == 502
    assert "does not support target language" in response.json()["detail"]
    assert list(tmp_path.glob("*/job.json")) == []


def test_capabilities_degrades_when_no_backend_answers(monkeypatch):
    import app

    monkeypatch.setattr(app, "_resolve_backend", lambda force=False: None)
    client = fastapi_testclient.TestClient(app.app)
    payload = client.get("/capabilities").json()
    assert payload["available"] is False
    assert payload["hint"]
    assert payload["stages"][0] == "extract"


def test_rejected_containers(service):
    client, _, _ = service
    response = client.post(
        "/jobs/upload",
        files={"file": ("clip.avi", b"not a video", "video/x-msvideo")},
        data={"target_language": "vi"},
    )
    assert response.status_code == 400


@ffmpeg
def test_the_history_starts_empty_and_then_lists_the_upload(service):
    client, _, tmp_path = service
    assert client.get("/jobs").json() == {"jobs": [], "total": 0}

    job_id = upload(client, tmp_path)
    payload = client.get("/jobs").json()
    assert payload["total"] == 1
    row = payload["jobs"][0]
    assert row["job_id"] == job_id
    assert row["source_filename"] == "source.mp4"
    assert row["target_language"] == "vi"
    assert row["created_at"], "a history list has to be orderable by more than mtime"
    assert row["progress"] == "1/11", "upload is the one stage that has run"
    assert row["download_url"] is None, "nothing is rendered yet"


@ffmpeg
def test_the_newest_job_is_listed_first(service):
    client, _, tmp_path = service
    first = upload(client, tmp_path)
    second = upload(client, tmp_path)
    listed = [row["job_id"] for row in client.get("/jobs").json()["jobs"]]
    assert listed[0] == second and first in listed


@ffmpeg
def test_progress_follows_the_stages_that_ran(service):
    client, _, tmp_path = service
    job_id = upload(client, tmp_path)
    stage(client, "extract", job_id)
    stage(client, "diarize", job_id)

    row = next(r for r in client.get("/jobs").json()["jobs"] if r["job_id"] == job_id)
    assert "extract" in row["completed_stages"]
    assert "diarize" in row["completed_stages"]
    assert row["progress"] == "3/11"
    assert 0 < row["percent"] <= 100


@ffmpeg
def test_a_failed_job_keeps_its_reason_in_the_list(service, monkeypatch):
    import app

    client, _, tmp_path = service
    job_id = upload(client, tmp_path)
    monkeypatch.setattr(app, "_run", lambda command: (_ for _ in ()).throw(RuntimeError("ffmpeg died")))
    client.post("/stages/extract", json={"job_id": job_id})

    row = next(r for r in client.get("/jobs").json()["jobs"] if r["job_id"] == job_id)
    assert row["status"] == "failed"
    assert row["error"]["stage"] == "extract"
    assert row["finished_at"], "a finished job is stamped whether it worked or not"


def test_an_unreadable_folder_does_not_hide_the_rest(service, tmp_path):
    client, _, _ = service
    (tmp_path / "abcdef123456").mkdir()
    (tmp_path / "abcdef123456" / "job.json").write_text("{ not json", encoding="utf-8")
    assert client.get("/jobs").json() == {"jobs": [], "total": 0}


@ffmpeg
def test_deleting_a_job_removes_it_and_its_files(service):
    client, _, tmp_path = service
    job_id = upload(client, tmp_path)
    assert (tmp_path / job_id).exists()

    assert client.delete(f"/jobs/{job_id}").json()["deleted"] is True
    assert not (tmp_path / job_id).exists()
    assert client.get("/jobs").json()["total"] == 0
    assert client.get(f"/jobs/{job_id}").status_code == 404


@ffmpeg
def test_a_finished_stage_is_reused_instead_of_re_run(service):
    client, calls, tmp_path = service
    job_id = upload(client, tmp_path)
    stage(client, "extract", job_id)
    stage(client, "transcribe", job_id)
    before = len([call for call in calls if call[0] == "/transcribe"])
    assert before == 1

    again = client.post("/stages/transcribe", json={"job_id": job_id}).json()
    assert again["reused"] is True
    assert again["status"] == "completed"
    after = len([call for call in calls if call[0] == "/transcribe"])
    assert after == before, "the backend must not be called again"


@ffmpeg
def test_force_re_runs_a_finished_stage(service):
    client, calls, tmp_path = service
    job_id = upload(client, tmp_path)
    stage(client, "extract", job_id)
    stage(client, "transcribe", job_id)

    response = client.post(
        "/stages/transcribe", json={"job_id": job_id, "force": True}
    ).json()
    assert response.get("reused") is None
    assert len([call for call in calls if call[0] == "/transcribe"]) == 2


@ffmpeg
def test_a_failed_job_resumes_at_the_stage_that_stopped(service, monkeypatch):
    import app

    client, calls, tmp_path = service
    job_id = upload(client, tmp_path)
    stage(client, "extract", job_id)
    stage(client, "transcribe", job_id)
    stage(client, "merge_segments", job_id)

    real = app._stage_call

    def broken(path, **kwargs):  # noqa: ANN001, ANN003
        if path == "/translate":
            raise RuntimeError("AI backend returned 500")
        return real(path, **kwargs)

    monkeypatch.setattr(app, "_stage_call", broken)
    assert client.post("/stages/translate", json={"job_id": job_id}).status_code == 500
    assert _load(tmp_path, job_id)["status"] == "failed"

    monkeypatch.setattr(app, "_stage_call", real)
    monkeypatch.setattr(app.httpx, "post", lambda *a, **k: _Accepted())
    started = client.post(f"/jobs/{job_id}/start").json()

    assert started["status"] == "accepted"
    assert started["resumed_from"] == "translate"
    assert "transcribe" in started["reusing"]
    assert "translate" not in started["reusing"], "the failed stage must run again"
    resumed = _load(tmp_path, job_id)
    assert resumed["status"] == "running"
    assert "error" not in resumed
    assert len(resumed["segments"]) == len(SEGMENTS)


@ffmpeg
def test_a_completed_job_is_not_restarted_by_accident(service, monkeypatch):
    import app

    client, _, tmp_path = service
    job_id = upload(client, tmp_path)
    for name in ["extract", "transcribe", "merge_segments"]:
        stage(client, name, job_id)
    monkeypatch.setattr(app.httpx, "post", lambda *a, **k: _Accepted())
    assert client.post(f"/jobs/{job_id}/start").json()["message"] == "Job has already started"


class _Accepted:
    status_code = 202

    def raise_for_status(self):  # noqa: ANN201
        return None


def _load(root, job_id):  # noqa: ANN001, ANN202
    return json.loads((root / job_id / "job.json").read_text(encoding="utf-8"))


@ffmpeg
def test_a_folder_this_build_would_not_name_can_still_be_deleted(service, tmp_path):
    client, _, _ = service
    stray = tmp_path / "fail0verte5t"
    stray.mkdir()
    (stray / "job.json").write_text(json.dumps({
        "job_id": "fail0verte5t", "status": "running", "completed_stages": ["upload"],
        "skipped_stages": [], "config": {}, "files": {}, "segments": [],
    }), encoding="utf-8")

    assert any(row["job_id"] == "fail0verte5t" for row in client.get("/jobs").json()["jobs"])
    assert client.delete("/jobs/fail0verte5t").json()["deleted"] is True
    assert not stray.exists()


def test_delete_refuses_to_escape_the_job_root(service, tmp_path):
    client, _, _ = service
    outsider = tmp_path.parent / "not-a-job"
    outsider.mkdir(exist_ok=True)
    for attempt in ("../not-a-job", "..", "a/b"):
        assert client.delete(f"/jobs/{attempt}").status_code in (404, 400, 405), attempt
    assert outsider.exists(), "nothing outside the job root may be removed"
