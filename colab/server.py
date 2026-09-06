"""HTTP API for notebook and local model inference."""
from __future__ import annotations

import json
import os
import secrets
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import providers  # noqa: E402
from core import config as job_config, feature_flags  # noqa: E402
from core.errors import InvalidRequest, ServiceError  # noqa: E402
from core.media import Scratch, duration, normalise_speech, sample_rate  # noqa: E402
from core.runtime import device, loaded, release_all  # noqa: E402
from dubflow_core import languages as L  # noqa: E402
from dubflow_core import segments as segment_tools  # noqa: E402
from fastapi import (  # noqa: E402
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from providers.asr import windows_from_turns  # noqa: E402
from providers.tts import SpeechRequest  # noqa: E402
from pydantic import BaseModel  # noqa: E402

import jobs  # noqa: E402
import pipeline  # noqa: E402
import tasks  # noqa: E402

AUTH_TOKEN = os.getenv("AUTH_TOKEN", "")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", providers.DEFAULTS["asr"])

app = FastAPI(title="DubFlow Colab AI", version="4.0")

@app.exception_handler(ServiceError)
def _service_error(_: Request, exc: ServiceError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


def _authorize(authorization: Optional[str]) -> None:
    if not AUTH_TOKEN:
        return
    if not secrets.compare_digest(authorization or "", f"Bearer {AUTH_TOKEN}"):
        raise HTTPException(status_code=401, detail="Invalid bearer token")


def _language(value: str, allow_auto: bool = False) -> Optional[str]:
    return job_config.as_language(value, "language", allow_auto=allow_auto)


def _persist(upload: UploadFile) -> Path:
    """Persist an upload before its request-scoped stream closes."""
    suffix = Path(upload.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        shutil.copyfileobj(upload.file, handle)
    return Path(handle.name)


def _deferred(work, async_mode: str, label: str):  # noqa: ANN001, ANN201
    """Run work now or return an asynchronous task."""
    if job_config.as_bool(async_mode, False):
        return JSONResponse(status_code=202, content=tasks.submit(work, label))
    return work()


@app.get("/tasks/{task_id}")
def read_task(task_id: str, authorization: Optional[str] = Header(None)) -> Dict:
    _authorize(authorization)
    return tasks.public(task_id)


@app.get("/tasks/{task_id}/download")
def download_task(task_id: str, authorization: Optional[str] = Header(None)):  # noqa: ANN201
    _authorize(authorization)
    payload = tasks.result_or_raise(task_id)
    if not payload or not payload.get("file"):
        raise ServiceError(f"Task '{task_id}' has no file to download", status_code=409)
    return FileResponse(
        payload["file"],
        media_type=payload.get("media_type", "application/octet-stream"),
        headers=payload.get("headers") or {},
    )


@app.get("/capabilities")
def capabilities(authorization: Optional[str] = Header(None)) -> Dict:
    """Return model metadata and runtime availability."""
    if AUTH_TOKEN and authorization:
        _authorize(authorization)
    payload = providers.capabilities()
    payload["stages"] = pipeline.STAGE_NAMES
    payload["optional_stages"] = pipeline.OPTIONAL_STAGES
    payload["version"] = app.version
    return payload


@app.get("/health")
def health() -> Dict:
    asr = providers.registry("asr")
    translation = providers.registry("translation")
    tts = providers.registry("tts")
    return {
        "status": "ready",
        "version": app.version,
        "device": device(),
        "services": ["transcription", "translation", "speech", "diarization", "pipeline"],
        "loaded": loaded(),
        "feature_flags": feature_flags.public(),
        "whisper_model": WHISPER_MODEL,
        "whisper_models": sorted(
            spec.id for spec in asr.for_provider("faster_whisper")
        ),
        "translation_model": translation.get("nllb").repo_id,
        "translation_engines": sorted(spec.id for spec in translation.models()),
        "tts_engines": {
            spec.id: {
                "available": spec.available(),
                "package": spec.optional_package,
            }
            for spec in tts.models()
        },
        "asr_models": [spec.id for spec in asr.models() if spec.available()],
        "capabilities_url": "/capabilities",
        "languages": L.listing(),
    }


@app.get("/languages")
def languages() -> Dict:
    return {"languages": L.listing()}


@app.post("/diarize")
def diarize(
    audio: UploadFile = File(...),
    provider: str = Form(""),
    model: str = Form(""),
    min_speakers: str = Form(""),
    max_speakers: str = Form(""),
    async_mode: str = Form(""),
    authorization: Optional[str] = Header(None),
):  # noqa: ANN201
    _authorize(authorization)
    spec = providers.find("diarization", provider or None,
                          model or providers.DEFAULTS["diarization"])
    providers.registry("diarization").require_available(spec)
    path = _persist(audio)
    lower = int(min_speakers) if min_speakers.strip() else None
    upper = int(max_speakers) if max_speakers.strip() else None

    def work() -> Dict:
        try:
            with jobs.lock:
                engine = providers.diarization.load(spec)
                turns = engine.diarize(path, min_speakers=lower, max_speakers=upper)
            return {
                "turns": turns,
                "speakers": segment_tools.speakers_of(
                    [{"speaker_id": turn["speaker_id"]} for turn in turns]
                ),
                "diarization_model": engine.name,
            }
        finally:
            path.unlink(missing_ok=True)

    return _deferred(work, async_mode, f"diarize:{spec.id}")


@app.post("/transcribe")
def transcribe(
    audio: UploadFile = File(...),
    language: str = Form("auto"),
    model: str = Form(""),
    provider: str = Form(""),
    turns: str = Form(""),
    async_mode: str = Form(""),
    authorization: Optional[str] = Header(None),
):  # noqa: ANN201
    _authorize(authorization)
    source = _language(language, allow_auto=True)
    spec = providers.find("asr", provider or None, model or WHISPER_MODEL)
    if source is None and not spec.supports_language_detection:
        raise InvalidRequest(
            f"'{spec.display_name}' cannot detect the spoken language. "
            f"Pass an explicit language."
        )
    if source is not None:
        providers.registry("asr").require_language(spec, source, "source")

    parsed = _parse_turns(turns)
    path = _persist(audio)

    def work() -> Dict:
        try:
            with jobs.lock:
                engine = providers.asr.load(spec, source)
                windows = (
                    windows_from_turns(parsed)
                    if parsed and not spec.supports_timestamps else None
                )
                result = engine.transcribe(path, source, windows)
            providers.registry("asr").require_language(spec, result.language, "source")

            segments = result.segments
            if parsed:
                segments = segment_tools.renumber(
                    segment_tools.assign_speakers(segments, parsed)
                )
            if not segments:
                raise ServiceError("No speech was found in the audio", status_code=422)
            return {
                "source_language": result.language,
                "segments": segments,
                "speakers": segment_tools.speakers_of(segments),
                "asr_provider": spec.provider,
                "asr_model": spec.id,
                "whisper_model": spec.id,
            }
        finally:
            path.unlink(missing_ok=True)

    return _deferred(work, async_mode, f"transcribe:{spec.id}")


def _parse_turns(raw: str) -> List[Dict]:
    if not raw or not raw.strip():
        return []
    try:
        parsed = json.loads(raw)
    except ValueError as exc:
        raise InvalidRequest("turns must be a JSON array of diarization turns") from exc
    if isinstance(parsed, dict):
        parsed = parsed.get("turns") or []
    if not isinstance(parsed, list):
        raise InvalidRequest("turns must be a JSON array of diarization turns")
    for turn in parsed:
        if not isinstance(turn, dict) or {"speaker_id", "start", "end"} - set(turn):
            raise InvalidRequest("each turn needs speaker_id, start and end")
    return parsed


class TranslationRequest(BaseModel):
    source_language: str
    target_language: str
    texts: List[str]
    engine: str = ""
    provider: str = ""
    model: str = ""
    async_mode: bool = False


@app.post("/translate")
def translate(
    request: TranslationRequest,
    authorization: Optional[str] = Header(None),
):  # noqa: ANN201
    _authorize(authorization)
    source = _language(request.source_language)
    target = _language(request.target_language)
    spec = providers.find(
        "translation",
        request.provider or None,
        request.model or request.engine or providers.DEFAULTS["translation"],
    )
    if not request.texts or len(request.texts) > 500:
        raise InvalidRequest("texts must contain 1 to 500 items")
    if source == target:
        return {
            "translations": request.texts,
            "translation_engine": spec.id,
            "translation_model": spec.id,
            "skipped": True,
        }
    registry = providers.registry("translation")
    registry.require_language(spec, source, "source")
    registry.require_language(spec, target, "target")
    texts = list(request.texts)

    def work() -> Dict:
        with jobs.lock:
            engine = providers.translation.load(spec)
            translations = engine.translate(texts, source, target)
        if len(translations) != len(texts):
            raise ServiceError("The translation model returned the wrong number of rows")
        return {
            "translations": [text.strip() for text in translations],
            "translation_engine": spec.id,
            "translation_model": spec.id,
            "translation_provider": spec.provider,
            "skipped": False,
        }

    return _deferred(work, "true" if request.async_mode else "", f"translate:{spec.id}")


@app.post("/synthesize")
def synthesize(
    text: str = Form(...),
    language: str = Form("vi"),
    speed: float = Form(1.0),
    engine: str = Form(""),
    provider: str = Form(""),
    model: str = Form(""),
    speaker_id: str = Form(""),
    multi_voice: str = Form(""),
    authorization: Optional[str] = Header(None),
) -> Response:
    _authorize(authorization)
    target = _language(language)
    requested_multi_voice = job_config.as_bool(multi_voice, False)
    if requested_multi_voice and not feature_flags.multi_voice_enabled():
        raise InvalidRequest(
            "Multi-voice is disabled. Start the AI service with "
            "DUBFLOW_MULTI_VOICE=true before requesting it."
        )
    default_model = "edge" if requested_multi_voice else providers.DEFAULTS["tts"]
    spec = providers.find("tts", provider or None, model or engine or default_model)
    if requested_multi_voice and not spec.supports_multispeaker:
        raise InvalidRequest(
            "Multi-voice mode requires Edge TTS. Restart without "
            "DUBFLOW_MULTI_VOICE=true to use a single-voice model."
        )
    providers.registry("tts").require_language(spec, target, "target")
    text = text.strip()
    if not text:
        raise InvalidRequest("Text is required")
    if len(text) > 2000:
        raise InvalidRequest("Text must be at most 2000 characters")
    if not 0.5 <= speed <= 2.0:
        raise InvalidRequest("Speed must be between 0.5 and 2.0")

    with Scratch(".wav") as output:
        with jobs.lock:
            engine_instance = providers.tts.load(spec, target)
            engine_instance.synthesize(
                SpeechRequest(
                    text=text,
                    language=target,
                    speed=speed,
                    speaker_id=speaker_id.strip() or None,
                    multi_voice=requested_multi_voice,
                ),
                output,
            )
        payload = output.read_bytes()
        if not payload:
            raise ServiceError(f"The '{spec.id}' engine produced no audio")
        rate = sample_rate(output)
        length = duration(output)

    headers = {
        "X-TTS-Model": engine_instance.name,
        "X-TTS-Engine": spec.id,
        "X-TTS-Provider": spec.provider,
        "X-TTS-Sample-Rate": str(rate),
        "X-TTS-Duration": f"{length:.3f}",
    }
    if engine_instance.selected_voice:
        headers["X-TTS-Voice"] = engine_instance.selected_voice

    return Response(
        content=payload,
        media_type="audio/wav",
        headers=headers,
    )


@app.post("/separate")
def separate(
    audio: UploadFile = File(...),
    provider: str = Form(""),
    model: str = Form(""),
    stem: str = Form("background"),
    async_mode: str = Form(""),
    authorization: Optional[str] = Header(None),
):  # noqa: ANN201
    _authorize(authorization)
    if stem not in {"background", "speech"}:
        raise InvalidRequest("stem must be 'background' or 'speech'")
    spec = providers.find("separation", provider or None,
                          model or providers.DEFAULTS["separation"])
    providers.registry("separation").require_available(spec)
    upload = _persist(audio)
    headers = {"X-Separation-Model": spec.id, "X-Separation-Stem": stem}
    deferred = job_config.as_bool(async_mode, False)

    def work() -> Dict:
        workdir = Path(tempfile.mkdtemp(prefix="dubflow-separate-"))
        try:
            source = workdir / f"input{upload.suffix}"
            shutil.copyfile(upload, source)
            with jobs.lock:
                engine = providers.separation.load(spec)
                stems = engine.separate(source, workdir)
            kept = Path(tempfile.mkdtemp(prefix="dubflow-stem-")) / f"{stem}.wav"
            shutil.move(str(stems[stem]), kept)
            return {"file": str(kept), "media_type": "audio/wav", "headers": headers}
        finally:
            upload.unlink(missing_ok=True)
            shutil.rmtree(workdir, ignore_errors=True)

    if deferred:
        state = tasks.submit(work, f"separate:{spec.id}")
        return JSONResponse(status_code=202, content=state)
    produced = work()
    return FileResponse(produced["file"], media_type="audio/wav", headers=headers)


class AlignmentRequest(BaseModel):
    segments: List[Dict[str, Any]]
    media_duration: Optional[float] = None
    min_speed: float = 0.0
    max_speed: float = 0.0
    allow_stretch: bool = False


@app.post("/align")
def align(request: AlignmentRequest, authorization: Optional[str] = Header(None)) -> Dict:
    _authorize(authorization)
    from dubflow_core import alignment

    limits = alignment.Limits(
        min_speed=request.min_speed or alignment.DEFAULT_MIN_SPEED,
        max_speed=request.max_speed or alignment.DEFAULT_MAX_SPEED,
        allow_stretch=request.allow_stretch,
    )
    plans = alignment.plan_segments(request.segments, request.media_duration, limits)
    return {
        "plans": [
            {
                "id": segment.get("id"),
                "speed": plan.speed,
                "status": plan.status,
                "window": plan.window,
                "overflow": plan.overflow,
            }
            for segment, plan in zip(request.segments, plans)
        ],
        "limits": {"min_speed": limits.min_speed, "max_speed": limits.max_speed},
    }


@app.post("/validate")
def validate(request: Dict[str, Any], authorization: Optional[str] = Header(None)) -> Dict:
    _authorize(authorization)
    return {"valid": True, "config": job_config.build(request).public()}


@app.post("/jobs", status_code=202)
async def create_job(
    request: Request,
    video: UploadFile = File(...),
    authorization: Optional[str] = Header(None),
) -> Dict:
    """Queue a complete dubbing job."""
    _authorize(authorization)
    form = await request.form()
    values = {
        name: value for name, value in form.items()
        if name != "video" and isinstance(value, str)
    }
    config = job_config.build(values)

    def save(target: Path) -> None:
        with target.open("wb") as handle:
            shutil.copyfileobj(video.file, handle, length=8 * 1024 * 1024)

    job = jobs.create(video.filename or "input.mp4", values, config.public(), save)
    ahead = jobs.enqueue(job["job_id"])
    return {**jobs.public(job), "queued_ahead": ahead}


@app.get("/jobs")
def list_jobs(authorization: Optional[str] = Header(None)) -> Dict:
    _authorize(authorization)
    return {"jobs": jobs.listing(), "queued": jobs.queued(), "limit": jobs.LIMIT}


@app.get("/jobs/{job_id}")
def get_job(job_id: str, authorization: Optional[str] = Header(None)) -> Dict:
    _authorize(authorization)
    return jobs.public(jobs.read(job_id))


@app.get("/jobs/{job_id}/segments")
def get_segments(job_id: str, authorization: Optional[str] = Header(None)) -> Dict:
    _authorize(authorization)
    job = jobs.read(job_id)
    return {
        "job_id": job_id,
        "segments": job.get("segments", []),
        "turns": job.get("turns", []),
    }


@app.get("/jobs/{job_id}/download")
def download_job(job_id: str, authorization: Optional[str] = Header(None)):  # noqa: ANN201
    _authorize(authorization)
    job = jobs.read(job_id)
    name = job.get("files", {}).get("output")
    if not name:
        raise ServiceError(
            f"Job '{job_id}' is {job['status']}, the video is not ready", status_code=409
        )
    return FileResponse(jobs.directory(job_id) / name, media_type="video/mp4", filename=name)


@app.get("/jobs/{job_id}/subtitle")
def download_subtitle(job_id: str, authorization: Optional[str] = Header(None)):  # noqa: ANN201
    _authorize(authorization)
    job = jobs.read(job_id)
    name = job.get("files", {}).get("subtitle")
    if not name:
        raise ServiceError("Subtitles are not ready", status_code=409)
    return FileResponse(
        jobs.directory(job_id) / name, media_type="application/x-subrip", filename=name
    )


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str, authorization: Optional[str] = Header(None)) -> Dict:
    _authorize(authorization)
    jobs.delete(job_id)
    return {"job_id": job_id, "deleted": True}


@app.post("/unload")
def unload(authorization: Optional[str] = Header(None)) -> Dict:
    _authorize(authorization)
    release_all()
    return {"loaded": loaded()}
