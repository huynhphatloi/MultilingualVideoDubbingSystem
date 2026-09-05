"""Small media API used by the n8n dubbing workflow.

The service deliberately has no database, object storage, frontend framework,
diarization, or model router. It also serves the single-file demo UI. Each job
is one directory containing a JSON manifest and the files produced by the seven
visible n8n stages. All model inference is delegated to Google Colab.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

ROOT = Path(os.getenv("DATA_ROOT", Path(__file__).resolve().parent.parent / "data/jobs"))
ROOT.mkdir(parents=True, exist_ok=True)

COLAB_API_URL = os.getenv("COLAB_API_URL", "").rstrip("/")
COLAB_API_TOKEN = os.getenv("COLAB_API_TOKEN", "")
COLAB_API_TIMEOUT = float(os.getenv("COLAB_API_TIMEOUT", "1800"))
#: Notebook backends to try, most preferred first. Each name NAME reads
#: NAME_API_URL and NAME_API_TOKEN, so the original COLAB_* pair still works
#: untouched. Neither Colab nor Kaggle can be started from here - both need a
#: human to press Run - so this picks a session that is already alive rather
#: than provisioning one.
AI_BACKENDS = [
    name.strip().lower()
    for name in os.getenv("AI_BACKENDS", "colab,kaggle").split(",")
    if name.strip()
]
#: Re-probing on every segment would add a round trip per TTS call.
BACKEND_PROBE_TTL = float(os.getenv("BACKEND_PROBE_TTL", "30"))
BACKEND_PROBE_TIMEOUT = float(os.getenv("BACKEND_PROBE_TIMEOUT", "10"))
N8N_WEBHOOK_URL = os.getenv(
    "N8N_WEBHOOK_URL", "http://n8n:5678/webhook/dubbing/start"
)

APP_DIR = Path(__file__).resolve().parent
FRONTEND_INDEX = Path(
    os.getenv("FRONTEND_INDEX", APP_DIR / "frontend" / "index.html")
)
if not FRONTEND_INDEX.exists():
    # Local development keeps frontend/ beside ai-service/ at repository root.
    FRONTEND_INDEX = APP_DIR.parent / "frontend" / "index.html"

# ISO-639-1 stays the public application code. Colab owns all model-specific
# language mappings.
LANGUAGES = {
    "en": "English", "vi": "Vietnamese", "ja": "Japanese",
    "ko": "Korean", "zh": "Chinese", "fr": "French", "de": "German",
    "es": "Spanish", "pt": "Portuguese", "it": "Italian", "ru": "Russian",
    "nl": "Dutch", "pl": "Polish", "tr": "Turkish", "ar": "Arabic",
    "hi": "Hindi", "id": "Indonesian", "th": "Thai", "cs": "Czech",
    "hu": "Hungarian", "uk": "Ukrainian", "ro": "Romanian",
    "sv": "Swedish", "da": "Danish", "fi": "Finnish", "no": "Norwegian",
    "el": "Greek", "he": "Hebrew", "ms": "Malay", "fa": "Persian",
    "bn": "Bengali", "ta": "Tamil", "te": "Telugu", "ur": "Urdu",
    "sw": "Swahili", "tl": "Tagalog", "km": "Khmer", "lo": "Lao",
    "my": "Burmese", "bg": "Bulgarian", "hr": "Croatian", "sr": "Serbian",
    "sk": "Slovak", "sl": "Slovenian", "ca": "Catalan", "hy": "Armenian",
    "ka": "Georgian", "ne": "Nepali", "si": "Sinhala", "mn": "Mongolian",
}

_JOB_ID = re.compile(r"^[a-f0-9]{12}$")
_ALLOWED_VIDEO = {".mp4", ".mkv", ".mov", ".webm", ".m4v"}
#: Whisper checkpoints the form may pick from. Bigger is slower but more
#: accurate; the ".en" and "distil-" ones only understand English.
_WHISPER_MODELS = {
    "tiny", "tiny.en", "base", "base.en", "small", "small.en",
    "medium", "medium.en", "large-v2", "large-v3", "large-v3-turbo",
    "distil-small.en", "distil-medium.en", "distil-large-v3",
}
_DEFAULT_WHISPER_MODEL = "small"
#: Translation models the Colab server keeps.
_TRANSLATION_ENGINES = {"nllb", "seamless"}
_DEFAULT_TRANSLATION_ENGINE = "nllb"
#: TTS engines the Colab server registers. Only "mms" is installed by the
#: notebook's base requirements; the rest need their own pip install there.
_TTS_ENGINES = {"mms", "edge", "piper", "xtts_v2", "vixtts", "f5_vi", "f5_base"}
_DEFAULT_TTS_ENGINE = "mms"
#: These clone a speaker instead of using a stock voice, so they refuse to
#: speak until a reference sample of that speaker is uploaded to Colab.
_CLONING_ENGINES = {"xtts_v2", "vixtts", "f5_vi", "f5_base"}
#: Cloning quality plateaus well before this; longer clips only cost upload time.
_REFERENCE_SECONDS = 12.0
app = FastAPI(title="Multilingual Dubbing API", version="2.0")


class JobRequest(BaseModel):
    job_id: str


def _job_dir(job_id: str) -> Path:
    if not _JOB_ID.fullmatch(job_id):
        raise HTTPException(400, "Invalid job_id")
    path = ROOT / job_id
    if not path.exists():
        raise HTTPException(404, f"Job '{job_id}' does not exist")
    return path


def _manifest_path(job_id: str) -> Path:
    return _job_dir(job_id) / "job.json"


def _load_job(job_id: str) -> dict:
    return json.loads(_manifest_path(job_id).read_text(encoding="utf-8"))


def _save_job(job: dict) -> None:
    path = _job_dir(job["job_id"]) / "job.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


@contextmanager
def _stage(job_id: str, name: str) -> Iterator[tuple[dict, Path]]:
    job = _load_job(job_id)
    folder = _job_dir(job_id)
    job["status"] = "running"
    job["current_stage"] = name
    _save_job(job)
    try:
        yield job, folder
    except Exception as exc:
        message = exc.detail if isinstance(exc, HTTPException) else str(exc)
        job["status"] = "failed"
        job["error"] = {"stage": name, "message": str(message)[:1000]}
        _save_job(job)
        if isinstance(exc, HTTPException):
            raise
        # A bare exception reaches n8n as an empty "Internal Server Error", so
        # the reason would only exist in job.json and the container log.
        raise HTTPException(500, f"Stage '{name}' failed: {message}"[:1000]) from exc
    else:
        job.setdefault("completed_stages", []).append(name)
        job["current_stage"] = None
        job["status"] = "completed" if name == "render" else "running"
        job.pop("error", None)
        _save_job(job)


def _run(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        detail = (result.stderr or result.stdout or "command failed")[-3000:]
        raise RuntimeError(detail)


def _duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError((result.stderr or "ffprobe failed")[-1000:])
    return float(result.stdout.strip())


#: The backend that answered most recently, plus when it was checked.
_live_backend: tuple[str, str, str] | None = None
_live_checked = 0.0


def _configured_backends() -> list[tuple[str, str, str]]:
    """Return (name, url, token) for every backend with a usable URL."""
    backends = []
    for name in AI_BACKENDS:
        url = os.getenv(f"{name.upper()}_API_URL", "").strip().rstrip("/")
        if url.startswith(("http://", "https://")):
            backends.append((name, url, os.getenv(f"{name.upper()}_API_TOKEN", "")))
    return backends


def _misconfigured_backends() -> list[str]:
    """Names whose URL is set but unusable, the classic URL/token swap."""
    broken = []
    for name in AI_BACKENDS:
        url = os.getenv(f"{name.upper()}_API_URL", "").strip()
        if url and not url.startswith(("http://", "https://")):
            broken.append(name)
    return broken


def _probe_backend(url: str, token: str) -> bool:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        response = httpx.get(
            f"{url}/health", headers=headers, timeout=BACKEND_PROBE_TIMEOUT
        )
    except httpx.RequestError:
        return False
    return response.status_code < 400


def _invalidate_backend() -> None:
    global _live_backend, _live_checked
    _live_backend = None
    _live_checked = 0.0


def _resolve_backend(force: bool = False) -> tuple[str, str, str] | None:
    """Pick the first configured backend that answers /health."""
    global _live_backend, _live_checked
    now = time.monotonic()
    if not force and _live_backend and now - _live_checked < BACKEND_PROBE_TTL:
        return _live_backend
    for backend in _configured_backends():
        if _probe_backend(backend[1], backend[2]):
            _live_backend = backend
            _live_checked = now
            return backend
    _live_backend = None
    _live_checked = now
    return None


def _backend_problem() -> str:
    """One message explaining why no notebook answered, and what to do."""
    configured = _configured_backends()
    broken = _misconfigured_backends()
    if broken:
        names = ", ".join(f"{name.upper()}_API_URL" for name in broken)
        return (
            f"{names} must start with http:// or https://. Check that the URL "
            f"and token were not swapped in .env."
        )
    if not configured:
        names = " or ".join(f"{name.upper()}_API_URL" for name in AI_BACKENDS)
        return (
            f"No AI backend is configured. Set {names} plus the matching "
            f"_API_TOKEN in .env, then run 'make restart'."
        )
    listed = ", ".join(f"{name} ({url})" for name, url, _ in configured)
    return (
        f"No configured AI backend answered /health: {listed}. Open the "
        f"notebook, run all cells, and put the printed URL in .env. A notebook "
        f"session cannot be started from here."
    )


def _colab_request(path: str, **kwargs):  # noqa: ANN003, ANN202
    """Call whichever notebook backend is alive. There is no local fallback."""
    backend = _resolve_backend()
    if backend is None:
        raise HTTPException(503, _backend_problem())

    # An upload stream cannot be replayed once a failed attempt has consumed
    # it, so only bodies we can rebuild are safe to retry elsewhere.
    repeatable = "files" not in kwargs
    attempted: list[str] = []
    while True:
        name, url, token = backend
        attempted.append(name)
        headers = dict(kwargs.get("headers") or {})
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            response = httpx.post(
                f"{url}{path}",
                timeout=COLAB_API_TIMEOUT,
                **{**kwargs, "headers": headers},
            )
        except httpx.RequestError as exc:
            _invalidate_backend()
            following = _resolve_backend(force=True) if repeatable else None
            if following is not None and following[0] not in attempted:
                backend = following
                continue
            raise HTTPException(
                503,
                f"AI backend '{name}' became unreachable: {str(exc)[:300]}. "
                f"{_backend_problem()}",
            ) from exc
        if response.status_code >= 400:
            raise HTTPException(
                502,
                f"AI backend '{name}' returned {response.status_code}: "
                f"{response.text[:500]}",
            )
        return response


def _srt_timestamp(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _write_subtitles(path: Path, segments: list[dict]) -> None:
    blocks = []
    for index, segment in enumerate(segments, start=1):
        blocks.extend(
            [
                str(index),
                f"{_srt_timestamp(segment['start'])} --> "
                f"{_srt_timestamp(segment['end'])}",
                segment.get("translated_text") or segment["source_text"],
                "",
            ]
        )
    path.write_text("\n".join(blocks), encoding="utf-8")


@app.get("/", include_in_schema=False)
def demo_frontend():  # noqa: ANN201
    if not FRONTEND_INDEX.exists():
        raise HTTPException(404, "Demo frontend is not installed")
    return FileResponse(FRONTEND_INDEX, media_type="text/html")


@app.get("/health")
def health() -> dict:
    # Cheap: reports the cached probe rather than dialling every backend.
    backend = _resolve_backend()
    return {
        "status": "ready",
        "version": "2.0",
        "jobs_root": str(ROOT),
        "languages": len(LANGUAGES),
        "ai_backend": backend[0] if backend else None,
        "ai_backend_url": backend[1] if backend else None,
        "colab_configured": bool(COLAB_API_URL),
        "local_models": False,
    }


@app.get("/backends")
def backends() -> dict:
    """Which notebook sessions are alive right now, in preference order.

    Neither Colab nor Kaggle can be started from here, so this reports what is
    already running instead of trying to bring a session up.
    """
    configured = _configured_backends()
    live = _resolve_backend(force=True)
    return {
        "order": AI_BACKENDS,
        "backends": [
            {
                "name": name,
                "url": url,
                "token": bool(token),
                "alive": _probe_backend(url, token),
                "active": bool(live) and live[0] == name,
            }
            for name, url, token in configured
        ],
        "active": live[0] if live else None,
        "hint": None if live else _backend_problem(),
    }


@app.get("/languages")
def languages() -> dict:
    return {
        "languages": [
            {"code": code, "name": name}
            for code, name in LANGUAGES.items()
        ]
    }


@app.post("/jobs/upload")
def upload_video(
    file: UploadFile = File(...),
    target_language: str = Form("vi"),
    source_language: str = Form("auto"),
    model: str = Form(_DEFAULT_WHISPER_MODEL),
    translation_engine: str = Form(_DEFAULT_TRANSLATION_ENGINE),
    tts_engine: str = Form(_DEFAULT_TTS_ENGINE),
) -> dict:
    target = target_language.strip().lower()
    source = source_language.strip().lower()
    source = None if source in {"", "auto"} else source
    whisper_model = model.strip().lower() or _DEFAULT_WHISPER_MODEL
    mt_engine = translation_engine.strip().lower() or _DEFAULT_TRANSLATION_ENGINE
    voice_engine = tts_engine.strip().lower() or _DEFAULT_TTS_ENGINE
    if target not in LANGUAGES:
        raise HTTPException(400, f"Unsupported target language '{target}'")
    if source and source not in LANGUAGES:
        raise HTTPException(400, f"Unsupported source language '{source}'")
    if whisper_model not in _WHISPER_MODELS:
        raise HTTPException(
            400,
            f"Unsupported Whisper model '{whisper_model}' "
            f"(pick one of {sorted(_WHISPER_MODELS)})",
        )
    if mt_engine not in _TRANSLATION_ENGINES:
        raise HTTPException(
            400,
            f"Unsupported translation engine '{mt_engine}' "
            f"(pick one of {sorted(_TRANSLATION_ENGINES)})",
        )
    if voice_engine not in _TTS_ENGINES:
        raise HTTPException(
            400,
            f"Unsupported voice engine '{voice_engine}' "
            f"(pick one of {sorted(_TTS_ENGINES)})",
        )
    suffix = Path(file.filename or "input.mp4").suffix.lower() or ".mp4"
    if suffix not in _ALLOWED_VIDEO:
        raise HTTPException(400, f"Unsupported video container '{suffix}'")

    job_id = uuid.uuid4().hex[:12]
    folder = ROOT / job_id
    folder.mkdir(parents=True)
    source_path = folder / f"input{suffix}"
    with source_path.open("wb") as output:
        shutil.copyfileobj(file.file, output, length=8 * 1024 * 1024)

    job = {
        "job_id": job_id,
        "status": "running",
        "current_stage": None,
        "completed_stages": ["upload"],
        "source_filename": file.filename,
        "source_language": source,
        "target_language": target,
        "whisper_model": whisper_model,
        "translation_engine": mt_engine,
        "tts_engine": voice_engine,
        "files": {"input": source_path.name},
        "segments": [],
    }
    (folder / "job.json").write_text(
        json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "job_id": job_id,
        "stage": "upload",
        "status": "completed",
        "target_language": target,
        "whisper_model": whisper_model,
        "translation_engine": mt_engine,
        "tts_engine": voice_engine,
    }


@app.post("/stages/extract")
def extract_audio(request: JobRequest) -> dict:
    with _stage(request.job_id, "extract") as (job, folder):
        video = folder / job["files"]["input"]
        original = folder / "original.wav"
        asr_audio = folder / "asr.wav"
        _run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(video),
            "-vn", "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", str(original),
        ])
        _run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(video),
            "-vn", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(asr_audio),
        ])
        job["duration_seconds"] = round(_duration(video), 3)
        job["files"].update({"original_audio": original.name, "asr_audio": asr_audio.name})
    return {
        "job_id": request.job_id,
        "stage": "extract",
        "status": "completed",
        "duration_seconds": job["duration_seconds"],
    }


@app.post("/stages/transcribe")
def transcribe(request: JobRequest) -> dict:
    with _stage(request.job_id, "transcribe") as (job, folder):
        audio = folder / job["files"]["asr_audio"]
        with audio.open("rb") as stream:
            response = _colab_request(
                "/transcribe",
                files={"audio": (audio.name, stream, "audio/wav")},
                data={
                    # An empty language means auto-detect. The Colab server
                    # forwards anything else straight to Whisper, which rejects
                    # the literal "auto" as an invalid language code.
                    "language": job.get("source_language") or "",
                    "model": job.get("whisper_model") or _DEFAULT_WHISPER_MODEL,
                },
            )
        try:
            payload = response.json()
            if not isinstance(payload, dict):
                raise TypeError("response is not an object")
            segments = payload.get("segments") or payload.get("windows")
            detected = payload.get("source_language") or payload.get("language")
        except (ValueError, KeyError, TypeError) as exc:
            raise RuntimeError("Colab returned an invalid transcription response") from exc
        if not isinstance(segments, list) or not segments:
            raise RuntimeError("Whisper found no speech in the video")
        for index, segment in enumerate(segments):
            segment["id"] = index
        if detected not in LANGUAGES:
            raise RuntimeError(f"Detected language '{detected}' is not configured")
        job["source_language"] = detected
        job["segments"] = segments
        (folder / "transcript.json").write_text(
            json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return {
        "job_id": request.job_id,
        "stage": "transcribe",
        "status": "completed",
        "source_language": job["source_language"],
        "whisper_model": job.get("whisper_model") or _DEFAULT_WHISPER_MODEL,
        "segment_count": len(job["segments"]),
    }


@app.post("/stages/translate")
def translate(request: JobRequest) -> dict:
    with _stage(request.job_id, "translate") as (job, folder):
        source = job["source_language"]
        target = job["target_language"]
        texts = [segment["source_text"] for segment in job["segments"]]

        if source == target:
            translations = texts
        else:
            response = _colab_request(
                "/translate",
                json={
                    "source_language": source,
                    "target_language": target,
                    "texts": texts,
                    "items": [{"text": text} for text in texts],
                    "engine": job.get("translation_engine")
                    or _DEFAULT_TRANSLATION_ENGINE,
                },
            )
            try:
                payload = response.json()
                if not isinstance(payload, dict):
                    raise TypeError("response is not an object")
                translations = payload.get("translations")
                if translations is None:
                    translations = [item["text"] for item in payload["results"]]
            except (ValueError, KeyError, TypeError) as exc:
                raise RuntimeError("Colab returned an invalid translation response") from exc
            if len(translations) != len(texts):
                raise RuntimeError("Colab returned the wrong number of translations")

        for segment, translated in zip(job["segments"], translations, strict=True):
            segment["translated_text"] = translated.strip()
        subtitle = folder / "translated.srt"
        _write_subtitles(subtitle, job["segments"])
        job["files"]["subtitle"] = subtitle.name
    return {
        "job_id": request.job_id,
        "stage": "translate",
        "status": "completed",
        "source_language": source,
        "target_language": target,
        "translation_engine": job.get("translation_engine")
        or _DEFAULT_TRANSLATION_ENGINE,
        "segment_count": len(job["segments"]),
    }


def _build_reference(folder: Path, job: dict) -> Path:
    """Cut a sample of the original speaker for the cloning engines.

    Taken from the first transcribed segment onwards, so the clip is known to
    hold speech rather than whatever silence or music opens the video.
    """
    original = folder / job["files"]["original_audio"]
    segments = job["segments"]
    if not segments:
        raise RuntimeError("A voice reference needs the transcript, which is empty")
    start = float(segments[0]["start"])
    span = float(segments[-1]["end"]) - start
    if span < 1.0:
        raise RuntimeError(
            "The video holds less than a second of speech, too little to clone a voice"
        )
    reference = folder / "reference.wav"
    _run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.3f}", "-t", f"{min(_REFERENCE_SECONDS, span):.3f}",
        "-i", str(original),
        "-vn", "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", str(reference),
    ])
    return reference


def _upload_reference(reference: Path) -> str:
    """Send the sample once per job; segments then cite it by id."""
    with reference.open("rb") as stream:
        response = _colab_request(
            "/reference", files={"audio": (reference.name, stream, "audio/wav")}
        )
    try:
        reference_id = response.json()["reference_id"]
    except (ValueError, KeyError, TypeError) as exc:
        raise RuntimeError("Colab returned no reference_id for the voice sample") from exc
    return reference_id


def _colab_speech(
    text: str,
    language: str,
    output: Path,
    engine: str = _DEFAULT_TTS_ENGINE,
    reference_id: str = "",
) -> str:
    # The engine has to be named: the Colab server's own default is a cloning
    # model, which refuses to speak without a reference sample.
    data = {"text": text, "language": language, "speed": "1.0", "engine": engine}
    if reference_id:
        data["reference_id"] = reference_id
    response = _colab_request("/synthesize", data=data)
    output.write_bytes(response.content)
    if not output.exists() or output.stat().st_size == 0:
        raise RuntimeError("Colab TTS returned an empty audio file")
    return response.headers.get("x-tts-model") or engine


@app.post("/stages/synthesize")
def synthesize(request: JobRequest) -> dict:
    with _stage(request.job_id, "synthesize") as (job, folder):
        target = job["target_language"]
        output_dir = folder / "tts"
        output_dir.mkdir(exist_ok=True)
        voice_engine = job.get("tts_engine") or _DEFAULT_TTS_ENGINE
        reference_id = ""
        if voice_engine in _CLONING_ENGINES:
            reference = _build_reference(folder, job)
            job["files"]["reference"] = reference.name
            reference_id = _upload_reference(reference)
        models: dict[str, int] = {}
        for segment in job["segments"]:
            text = (segment.get("translated_text") or "").strip()
            if not text:
                continue
            output = output_dir / f"{segment['id']:04d}.wav"
            model_name = _colab_speech(
                text, target, output, voice_engine, reference_id
            )
            segment["tts_file"] = str(output.relative_to(folder))
            segment["tts_duration"] = round(_duration(output), 3)
            segment["tts_model"] = model_name
            models[model_name] = models.get(model_name, 0) + 1
        job["tts_provider"] = "colab"

    generated = sum(1 for segment in job["segments"] if segment.get("tts_file"))
    return {
        "job_id": request.job_id,
        "stage": "synthesize",
        "status": "completed",
        "generated_segments": generated,
        "requested_engine": voice_engine,
        "voice_reference": bool(reference_id),
        "provider": job["tts_provider"],
        "models_used": models,
    }


@app.post("/stages/mix")
def mix(request: JobRequest) -> dict:
    with _stage(request.job_id, "mix") as (job, folder):
        voiced = [segment for segment in job["segments"] if segment.get("tts_file")]
        if not voiced:
            raise RuntimeError("No generated speech is available")

        dubbed = folder / "dubbed.wav"
        command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
        for segment in voiced:
            command.extend(["-i", str(folder / segment["tts_file"])])

        filters = []
        labels = []
        for index, segment in enumerate(voiced):
            delay = max(0, round(float(segment["start"]) * 1000))
            label = f"s{index}"
            filters.append(
                f"[{index}:a]aresample=48000,asetpts=PTS-STARTPTS,"
                f"adelay={delay}|{delay}[{label}]"
            )
            labels.append(f"[{label}]")
        duration = float(job["duration_seconds"])
        filters.append(
            f"{''.join(labels)}amix=inputs={len(labels)}:duration=longest:normalize=0,"
            f"apad=whole_dur={duration:.3f},atrim=0:{duration:.3f}[dub]"
        )
        command.extend([
            "-filter_complex", ";".join(filters), "-map", "[dub]",
            "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", str(dubbed),
        ])
        _run(command)

        # The simple pipeline intentionally skips Demucs. The original soundtrack is kept at
        # low volume, producing a simple voice-over rather than a studio dub.
        final_audio = folder / "final.wav"
        _run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(folder / job["files"]["original_audio"]), "-i", str(dubbed),
            "-filter_complex",
            "[0:a]volume=0.25[background];[1:a]volume=1.5[voice];"
            "[background][voice]amix=inputs=2:duration=first:normalize=0,"
            "loudnorm=I=-16:TP=-1.5:LRA=11[out]",
            "-map", "[out]", "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le",
            str(final_audio),
        ])
        job["files"].update({"dubbed_audio": dubbed.name, "final_audio": final_audio.name})
    return {
        "job_id": request.job_id,
        "stage": "mix",
        "status": "completed",
        "mode": "voice-over",
        "placed_segments": len(voiced),
    }


@app.post("/stages/render")
def render(request: JobRequest) -> dict:
    with _stage(request.job_id, "render") as (job, folder):
        video = folder / job["files"]["input"]
        audio = folder / job["files"]["final_audio"]
        subtitle = folder / job["files"]["subtitle"]
        output = folder / f"dubbed_{job['target_language']}.mp4"
        duration = float(job["duration_seconds"])
        base = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(video), "-i", str(audio), "-i", str(subtitle),
            "-map", "0:v:0", "-map", "1:a:0", "-map", "2:0",
        ]
        finish = [
            "-c:a", "aac", "-b:a", "192k", "-c:s", "mov_text",
            "-metadata:s:s:0", f"language={job['target_language']}",
            "-t", f"{duration:.3f}", "-movflags", "+faststart", str(output),
        ]
        try:
            _run(base + ["-c:v", "copy"] + finish)
        except RuntimeError:
            _run(base + ["-c:v", "libx264", "-preset", "fast", "-crf", "22"] + finish)
        job["files"]["output"] = output.name
        job["output_url"] = f"{os.getenv('PUBLIC_BASE_URL', 'http://localhost:8000')}" \
                            f"/jobs/{request.job_id}/download"
    return {
        "job_id": request.job_id,
        "stage": "render",
        "status": "completed",
        "output_url": job["output_url"],
        "subtitle_url": f"{os.getenv('PUBLIC_BASE_URL', 'http://localhost:8000')}"
                        f"/jobs/{request.job_id}/subtitle",
    }


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    return _load_job(job_id)


@app.post("/jobs/{job_id}/start", status_code=202)
def start_job(job_id: str) -> dict:
    """Ask n8n to orchestrate the six stages after upload."""
    job = _load_job(job_id)
    completed = job.get("completed_stages", [])
    if job.get("status") == "completed" or len(completed) > 1:
        return {
            "job_id": job_id,
            "status": job.get("status", "running"),
            "message": "Job has already started",
        }

    import httpx

    try:
        response = httpx.post(
            N8N_WEBHOOK_URL,
            json={"job_id": job_id},
            timeout=30,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(
            503,
            "n8n did not accept the job. Open http://localhost:5678 and "
            "activate 'Multilingual Dubbing', then try again. "
            f"Technical detail: {str(exc)[:300]}",
        ) from exc

    return {"job_id": job_id, "status": "accepted", "orchestrator": "n8n"}


@app.get("/jobs/{job_id}/download")
def download(job_id: str):  # noqa: ANN201
    job = _load_job(job_id)
    output = _job_dir(job_id) / job.get("files", {}).get("output", "missing")
    if not output.exists():
        raise HTTPException(404, "Rendered video is not available yet")
    return FileResponse(output, media_type="video/mp4", filename=output.name)


@app.get("/jobs/{job_id}/subtitle")
def download_subtitle(job_id: str):  # noqa: ANN201
    job = _load_job(job_id)
    subtitle = _job_dir(job_id) / job.get("files", {}).get("subtitle", "missing")
    if not subtitle.exists():
        raise HTTPException(404, "Subtitle is not available yet")
    return FileResponse(subtitle, media_type="application/x-subrip", filename=subtitle.name)
