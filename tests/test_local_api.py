"""The local FFmpeg service: its stages, with the AI backend stubbed out.

The n8n route runs a second copy of the pipeline - extract, merge, references,
align, mix and render happen locally and only the model calls cross the tunnel.
Those local stages are exercised here against a real video; every HTTP call to
the notebook is replaced, so no model is downloaded and no GPU is needed.
"""
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
    "tts": {"provider": "mms", "model": "mms", "voice_cloning": False},
    "features": {"diarization": False, "voice_cloning": False, "alignment": True,
                 "source_separation": False},
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
    #: Model call -> canned answer. Only these cross the tunnel in production.
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
        if path == "/reference":
            return FakeResponse({"reference_id": "ref-" + str(len(calls))})
        if path == "/synthesize":
            source = kwargs["data"]["text"].split("] ", 1)[-1]
            return FakeResponse(
                content=tone_bytes(speech.get(source, 1.0)),
                headers={"x-tts-model": "test/tts"},
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
    client, calls, tmp_path = service
    job_id = upload(client, tmp_path)

    results = {name: stage(client, name, job_id) for name in [
        "extract", "diarize", "transcribe", "merge_segments", "translate",
        "voice_references", "synthesize", "align", "separate", "mix", "render",
    ]}

    # Optional stages skip themselves rather than failing the workflow, which is
    # what lets one n8n graph serve every configuration.
    for optional in ("diarize", "voice_references", "separate"):
        assert results[optional]["status"] == "skipped", optional
    assert results["mix"]["mode"] == "voice-over"

    job = client.get(f"/jobs/{job_id}").json()
    assert job["status"] == "completed"
    assert job["completed_stages"] == [
        "upload", "extract", "transcribe", "merge_segments", "translate",
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
def test_diarization_and_per_speaker_references(service, monkeypatch):
    client, calls, tmp_path = service
    import app

    diarized = json.loads(json.dumps(CONFIG))
    diarized["features"]["diarization"] = True
    diarized["features"]["voice_cloning"] = True
    diarized["tts"] = {"provider": "f5", "model": "f5_vi", "voice_cloning": True}
    diarized["diarization"] = {"provider": "pyannote", "model": "pyannote_3_1"}
    original = app._colab_request

    def routed(path, method="POST", **kwargs):  # noqa: ANN001, ANN202
        if path == "/validate":
            return FakeResponse({"valid": True, "config": diarized})
        return original(path, method, **kwargs)

    monkeypatch.setattr(app, "_colab_request", routed)

    job_id = upload(client, tmp_path, enable_diarization="true")
    for name in ("extract", "diarize", "transcribe", "merge_segments", "translate",
                 "voice_references", "synthesize"):
        stage(client, name, job_id)

    job = client.get(f"/jobs/{job_id}").json()
    assert job["speakers"] == ["SPEAKER_00", "SPEAKER_01"]
    assert [segment["speaker_id"] for segment in job["segments"]] == [
        "SPEAKER_00", "SPEAKER_01"
    ]
    references = job["references"]
    assert set(references) == {"SPEAKER_00", "SPEAKER_01"}
    for speaker, entry in references.items():
        assert (tmp_path / job_id / entry["file"]).exists()
        assert entry["reference_id"]
        assert entry["text"]

    # Each segment was voiced with its own speaker's reference id.
    used = [
        payload["reference_id"]
        for path, payload in calls
        if path == "/synthesize" and "reference_id" in payload
    ]
    assert len(set(used)) == 2


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
            data={"target_language": "vi", "tts_model": "f5_base"},
        )
    assert response.status_code == 502
    assert "does not support target language" in response.json()["detail"]
    # Nothing was stored for a job that could never run.
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
    client, _, tmp_path = service
    response = client.post(
        "/jobs/upload",
        files={"file": ("clip.avi", b"not a video", "video/x-msvideo")},
        data={"target_language": "vi"},
    )
    assert response.status_code == 400


# ==========================================================================
# Job history
# ==========================================================================
@ffmpeg
def test_the_history_starts_empty_and_then_lists_the_upload(service):
    """The browser used to hold the only record of a job id, so a reload lost
    the run and a second upload replaced the first in the panel."""
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
    assert row["progress"] == "1/12", "upload is the one stage that has run"
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
    stage(client, "diarize", job_id)  # optional, off for this job: skips itself

    row = next(r for r in client.get("/jobs").json()["jobs"] if r["job_id"] == job_id)
    assert "extract" in row["completed_stages"]
    assert "diarize" in row["skipped_stages"]
    # A skipped stage leaves the denominator, so the bar cannot exceed itself.
    assert row["progress"] == "2/11"
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
    """An upload interrupted midway leaves a folder without a usable manifest.
    One of those must not take the whole history down with it."""
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


# ==========================================================================
# Resuming a failed job
# ==========================================================================
@ffmpeg
def test_a_finished_stage_is_reused_instead_of_re_run(service, monkeypatch):
    """Transcription is the expensive stage. Re-running the workflow must not
    pay for it twice just because a later stage failed."""
    import app

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
    """Changing a model is a reason to run a stage again on purpose."""
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

    # Translate fails the way the real one did: a 500 from the AI backend.
    real = app._stage_call

    def broken(path, **kwargs):  # noqa: ANN001, ANN003
        if path == "/translate":
            raise RuntimeError("AI backend returned 500")
        return real(path, **kwargs)

    monkeypatch.setattr(app, "_stage_call", broken)
    assert client.post("/stages/translate", json={"job_id": job_id}).status_code == 500
    assert _load(tmp_path, job_id)["status"] == "failed"

    # The operator fixes the backend and presses retry.
    monkeypatch.setattr(app, "_stage_call", real)
    monkeypatch.setattr(app.httpx, "post", lambda *a, **k: _Accepted())
    started = client.post(f"/jobs/{job_id}/start").json()

    assert started["status"] == "accepted"
    assert started["resumed_from"] == "translate"
    assert "transcribe" in started["reusing"]
    assert "translate" not in started["reusing"], "the failed stage must run again"
    # The job is no longer failed, and the transcript it already produced is
    # still there - which is the whole point of resuming rather than re-uploading.
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
