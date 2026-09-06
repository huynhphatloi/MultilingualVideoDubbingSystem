"""HTTP API for notebook and local model inference."""
from __future__ import annotations

import json
import os
import secrets
import shutil
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import providers  # noqa: E402
from core import config as job_config  # noqa: E402
from core.errors import InvalidRequest, ServiceError  # noqa: E402
from core.media import Scratch, duration, sample_rate  # noqa: E402
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
from starlette.background import BackgroundTask  # noqa: E402

import tasks  # noqa: E402

AUTH_TOKEN = os.getenv("AUTH_TOKEN", "")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", providers.DEFAULTS["asr"])
MODEL_LOCK = threading.RLock()

app = FastAPI(title="DubFlow Colab AI", version="5.0")


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
    _authorize(authorization)
    payload = providers.capabilities()
    payload["version"] = app.version
    return payload


@app.get("/health")
def health() -> Dict:
    return {
        "status": "ready",
        "version": app.version,
        "device": device(),
        "services": [
            "transcription",
            "translation",
            "speech",
            "diarization",
            "separation",
        ],
        "loaded": loaded(),
    }


@app.get("/languages")
def languages(authorization: Optional[str] = Header(None)) -> Dict:
    _authorize(authorization)
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
    try:
        lower = int(min_speakers) if min_speakers.strip() else None
        upper = int(max_speakers) if max_speakers.strip() else None
    except ValueError as exc:
        raise InvalidRequest("min_speakers and max_speakers must be whole numbers") from exc
    for name, count in (("min_speakers", lower), ("max_speakers", upper)):
        if count is not None and not 1 <= count <= 20:
            raise InvalidRequest(f"{name} must be between 1 and 20")
    if lower is not None and upper is not None and lower > upper:
        raise InvalidRequest("min_speakers cannot be greater than max_speakers")
    path = _persist(audio)

    def work() -> Dict:
        try:
            with MODEL_LOCK:
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
            with MODEL_LOCK:
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
        request.model or providers.DEFAULTS["translation"],
    )
    if not request.texts or len(request.texts) > 500:
        raise InvalidRequest("texts must contain 1 to 500 items")
    if source == target:
        return {
            "translations": request.texts,
            "translation_model": spec.id,
            "skipped": True,
        }
    registry = providers.registry("translation")
    registry.require_language(spec, source, "source")
    registry.require_language(spec, target, "target")
    texts = list(request.texts)

    def work() -> Dict:
        with MODEL_LOCK:
            engine = providers.translation.load(spec)
            translations = engine.translate(texts, source, target)
        if len(translations) != len(texts):
            raise ServiceError("The translation model returned the wrong number of rows")
        return {
            "translations": [text.strip() for text in translations],
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
    provider: str = Form(""),
    model: str = Form(""),
    speaker_id: str = Form(""),
    authorization: Optional[str] = Header(None),
) -> Response:
    _authorize(authorization)
    target = _language(language)
    spec = providers.find("tts", provider or None, model or providers.DEFAULTS["tts"])
    providers.registry("tts").require_language(spec, target, "target")
    text = text.strip()
    if not text:
        raise InvalidRequest("Text is required")
    if len(text) > 2000:
        raise InvalidRequest("Text must be at most 2000 characters")
    if not 0.5 <= speed <= 2.0:
        raise InvalidRequest("Speed must be between 0.5 and 2.0")

    with Scratch(".wav") as output:
        with MODEL_LOCK:
            engine_instance = providers.tts.load(spec, target)
            engine_instance.synthesize(
                SpeechRequest(
                    text=text,
                    language=target,
                    speed=speed,
                    speaker_id=speaker_id.strip() or None,
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
            with MODEL_LOCK:
                engine = providers.separation.load(spec)
                stems = engine.separate(source, workdir)
            with tempfile.NamedTemporaryFile(suffix=f"-{stem}.wav", delete=False) as handle:
                kept = Path(handle.name)
            shutil.move(str(stems[stem]), kept)
            return {"file": str(kept), "media_type": "audio/wav", "headers": headers}
        finally:
            upload.unlink(missing_ok=True)
            shutil.rmtree(workdir, ignore_errors=True)

    if deferred:
        state = tasks.submit(work, f"separate:{spec.id}")
        return JSONResponse(status_code=202, content=state)
    produced = work()
    return FileResponse(
        produced["file"],
        media_type="audio/wav",
        headers=headers,
        background=BackgroundTask(Path(produced["file"]).unlink, missing_ok=True),
    )


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


@app.post("/unload")
def unload(authorization: Optional[str] = Header(None)) -> Dict:
    _authorize(authorization)
    with MODEL_LOCK:
        release_all()
    return {"loaded": loaded()}
