from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

import jobs
import pipeline
import providers
from core import config as job_config
from providers.asr.base import ASRProvider, Transcript
from providers.tts.base import TTSProvider

ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg is required for the pipeline test",
)

SCRIPT = [
    ("SPEAKER_00", 0.5, 3.0, "Good morning everyone", 2.0),
    ("SPEAKER_01", 3.5, 6.5, "Thank you for having me", 5.4),
    ("SPEAKER_00", 7.0, 9.5, "Let us begin", 1.5),
]
TURNS = [
    {"speaker_id": "SPEAKER_00", "start": 0.4, "end": 3.1},
    {"speaker_id": "SPEAKER_01", "start": 3.4, "end": 6.6},
    {"speaker_id": "SPEAKER_00", "start": 6.9, "end": 9.6},
]


def make_video(path: Path, seconds: float = 11.0) -> Path:
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", f"testsrc=duration={seconds}:size=320x240:rate=10",
            "-f", "lavfi", "-i", f"sine=frequency=300:duration={seconds}",
            "-c:v", "mpeg4", "-c:a", "aac", "-shortest", str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def tone(path: Path, seconds: float, frequency: int = 440) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", f"sine=frequency={frequency}:duration={seconds:.3f}",
            "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", str(path),
        ],
        check=True,
        capture_output=True,
    )


class FakeASR(ASRProvider):
    name = "test/asr"
    seen = []

    def detect_language(self, audio):  # noqa: ANN001, ANN201
        FakeASR.seen.append("detect")
        return "en"

    def transcribe(self, audio, language, windows=None):  # noqa: ANN001, ANN201
        from dubflow_core.segments import make_segment

        FakeASR.seen.append(("transcribe", language))
        segments = [
            make_segment(index, start, end, text)
            for index, (_, start, end, text, _) in enumerate(SCRIPT)
        ]
        return Transcript(language or "en", segments)


class FakeTranslation:
    name = "test/translation"

    def translate(self, texts, source, target):  # noqa: ANN001, ANN201
        return [f"[{target}] {text}" for text in texts]


class FakeTTS(TTSProvider):
    name = "test/tts"

    def __init__(self, spec):  # noqa: ANN001
        super().__init__(spec)
        self.calls = []

    def synthesize(self, request, out):  # noqa: ANN001, ANN201
        lengths = {text: seconds for _, _, _, text, seconds in SCRIPT}
        source = request.text.split("] ", 1)[-1]
        self.calls.append(request.speaker_id)
        tone(out, lengths.get(source, 1.0))


class FakeDiarization:
    name = "test/diarization"

    def diarize(self, audio, min_speakers=None, max_speakers=None):  # noqa: ANN001, ANN201
        return list(TURNS)


@pytest.fixture
def stub_models(monkeypatch):  # noqa: ANN201
    voice = {}

    def load_tts(spec, language=None):  # noqa: ANN001, ANN202
        voice.setdefault("engine", FakeTTS(spec))
        return voice["engine"]

    monkeypatch.setattr(providers.asr, "load", lambda spec, language=None: FakeASR(spec))
    monkeypatch.setattr(providers.translation, "load", lambda spec: FakeTranslation())
    monkeypatch.setattr(providers.tts, "load", load_tts)
    monkeypatch.setattr(providers.diarization, "load", lambda spec: FakeDiarization())
    from providers.base import ModelSpec

    monkeypatch.setattr(ModelSpec, "installed", lambda self: True)
    monkeypatch.setattr(ModelSpec, "credential_present", lambda self: True)
    return voice


def run_job(tmp_path, monkeypatch, values):  # noqa: ANN001, ANN201
    monkeypatch.setattr(jobs, "ROOT", tmp_path)
    video = make_video(tmp_path / "source.mp4")
    config = job_config.build(values)
    job = jobs.create("source.mp4", values, config.public(),
                      lambda target: shutil.copyfile(video, target))
    jobs.process(job["job_id"])
    return jobs.read(job["job_id"]), jobs.directory(job["job_id"])


@ffmpeg
def test_the_default_pipeline_produces_a_video(tmp_path, monkeypatch, stub_models):
    job, folder = run_job(tmp_path, monkeypatch, {
        "target_language": "vi", "source_language": "en", "tts_model": "mms",
    })

    assert job["status"] == "completed", job.get("error")
    assert job["completed_stages"] == [
        "extract", "transcribe", "merge_segments", "translate", "synthesize",
        "align", "mix", "render",
    ]
    assert set(job["skipped_stages"]) == {"diarize", "separate"}

    output = folder / job["files"]["output"]
    assert output.exists() and output.stat().st_size > 0
    assert (folder / job["files"]["subtitle"]).read_text(encoding="utf-8").startswith("1\n")
    assert job["speakers"] == ["SPEAKER_00"]
    assert job["mix_mode"] == "voice-over"


@ffmpeg
def test_alignment_shortens_the_line_that_would_have_overrun(tmp_path, monkeypatch, stub_models):
    job, folder = run_job(tmp_path, monkeypatch, {
        "target_language": "vi", "source_language": "en", "tts_model": "mms",
    })
    by_text = {segment["source_text"]: segment for segment in job["segments"]}

    short = by_text["Good morning everyone"]
    assert short["alignment_status"] == "fits"
    assert short["alignment_speed"] == 1.0

    long = by_text["Thank you for having me"]
    assert long["tts_duration_raw"] == pytest.approx(5.4, abs=0.2)
    assert long["alignment_status"] in {"aligned", "clamped"}
    assert long["alignment_speed"] > 1.0
    assert long["tts_duration_final"] < long["tts_duration_raw"]
    assert (folder / long["tts_file"]).exists()
    assert set(job["alignment"]) <= {"fits", "aligned", "clamped", "unmeasured"}


@ffmpeg
def test_alignment_can_be_switched_off(tmp_path, monkeypatch, stub_models):
    job, _ = run_job(tmp_path, monkeypatch, {
        "target_language": "vi", "source_language": "en", "tts_model": "mms",
        "enable_alignment": "false",
    })
    assert "align" in job["skipped_stages"]
    assert all("alignment_status" not in segment for segment in job["segments"])


@ffmpeg
def test_diarization_carries_the_speaker_through_to_the_voice(tmp_path, monkeypatch, stub_models):
    """No engine in this build varies its voice by speaker, so the dub sounds
    the same either way. What diarization still does is decide who owns each
    line, and that label reaches the TTS request - which is the hook a
    multi-voice provider would use. See "Future work" in the README."""
    job, _ = run_job(tmp_path, monkeypatch, {
        "target_language": "vi", "source_language": "en", "tts_model": "mms",
        "enable_diarization": "true",
    })

    assert job["status"] == "completed", job.get("error")
    assert job["speakers"] == ["SPEAKER_00", "SPEAKER_01"]
    assert [segment["speaker_id"] for segment in job["segments"]] == [
        "SPEAKER_00", "SPEAKER_01", "SPEAKER_00"
    ]
    assert set(stub_models["engine"].calls) == {"SPEAKER_00", "SPEAKER_01"}


@ffmpeg
def test_a_failing_stage_records_where_it_stopped(tmp_path, monkeypatch, stub_models):
    def explode(job, folder):  # noqa: ANN001
        raise RuntimeError("the translation model fell over")

    monkeypatch.setattr(pipeline.translate, "run", explode)
    job, _ = run_job(tmp_path, monkeypatch, {
        "target_language": "vi", "source_language": "en", "tts_model": "mms",
    })
    assert job["status"] == "failed"
    assert job["error"]["stage"] == "translate"
    assert "fell over" in job["error"]["message"]
    assert job["completed_stages"] == ["extract", "transcribe", "merge_segments"]


@ffmpeg
def test_translation_is_skipped_when_source_and_target_match(tmp_path, monkeypatch, stub_models):
    job, _ = run_job(tmp_path, monkeypatch, {
        "target_language": "en", "source_language": "en", "tts_model": "mms",
    })
    assert job["status"] == "completed", job.get("error")
    assert job["translation_skipped"] is True
    assert job["segments"][0]["translated_text"] == job["segments"][0]["source_text"]


@ffmpeg
def test_public_status_reports_progress_against_the_planned_stages(tmp_path, monkeypatch, stub_models):
    job, _ = run_job(tmp_path, monkeypatch, {
        "target_language": "vi", "source_language": "en", "tts_model": "mms",
    })
    public = jobs.public(job)
    assert public["progress"] == "8/8"
    assert public["status"] == "completed"
    assert public["download_url"].endswith("/download")
    assert public["whisper_model"] == "small"
    assert public["translation_engine"] == "nllb"
    assert public["tts_engine"] == "mms"
    assert public["config"]["features"]["alignment"] is True


@ffmpeg
def test_language_detection_is_not_a_second_pass(tmp_path, monkeypatch, stub_models):
    FakeASR.seen = []
    job, _ = run_job(tmp_path, monkeypatch, {
        "target_language": "vi", "tts_model": "mms",
    })
    assert job["status"] == "completed", job.get("error")
    assert FakeASR.seen == [("transcribe", None)]
    assert job["source_language"] == "en"
    assert job["detected_language"] == "en"
