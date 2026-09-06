"""Google Colab AI backend: HTTP only.

Every model lives behind a provider (colab/providers) and every stage behind a
pipeline module (colab/pipeline). This file does authentication, request
parsing, and turning a ServiceError into an HTTP response - nothing else. That
is why it no longer imports torch, transformers or faster-whisper: a session
where none of the optional engines are installed still starts, still answers
/capabilities, and reports each engine as unavailable instead of crashing.

Two routes into the same pipeline:

* the stage endpoints (/transcribe, /translate, /synthesize, /diarize, /align)
  which n8n orchestrates and which any other application can reuse;
* POST /jobs, which runs all eleven stages here and needs nothing but this URL.
"""
from __future__ import annotations

import json
import os
import secrets
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

#: The shared, dependency-free package lives beside colab/ in the repository.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import providers  # noqa: E402
from core import config as job_config  # noqa: E402
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

AUTH_TOKEN = os.getenv("AUTH_TOKEN", "")
#: Only the fallback when a request omits the model.
WHISPER_MODEL = os.getenv("WHISPER_MODEL", providers.DEFAULTS["asr"])

app = FastAPI(title="DubFlow Colab AI", version="4.0")

_REFERENCE_DIR = Path(tempfile.gettempdir()) / "dubflow-references"
_REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
_REFERENCE_LIMIT = 16
#: A reference clip may carry the transcript the cloning engines want.
_REFERENCE_TEXT = _REFERENCE_DIR / "texts.json"


@app.exception_handler(ServiceError)
def _service_error(_: Request, exc: ServiceError) -> JSONResponse:
    """One place where a provider's error becomes an HTTP response."""
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


def _authorize(authorization: Optional[str]) -> None:
    """Bearer check. The tunnel is public, so the comparison is constant-time."""
    if not AUTH_TOKEN:
        return
    if not secrets.compare_digest(authorization or "", f"Bearer {AUTH_TOKEN}"):
        raise HTTPException(status_code=401, detail="Invalid bearer token")


def _language(value: str, allow_auto: bool = False) -> Optional[str]:
    return job_config.as_language(value, "language", allow_auto=allow_auto)


# ==========================================================================
# Capabilities
# ==========================================================================
@app.get("/capabilities")
def capabilities(authorization: Optional[str] = Header(None)) -> Dict:
    """Everything this session offers, and what it can actually run.

    The frontend and n8n read this instead of keeping their own model lists,
    which is what used to let the three drift apart.
    """
    if AUTH_TOKEN and authorization:
        _authorize(authorization)
    payload = providers.capabilities()
    payload["stages"] = pipeline.STAGE_NAMES
    payload["optional_stages"] = pipeline.OPTIONAL_STAGES
    payload["version"] = app.version
    return payload


@app.get("/health")
def health() -> Dict:
    """Liveness, plus the pre-4.0 shape so existing callers keep working."""
    asr = providers.registry("asr")
    translation = providers.registry("translation")
    tts = providers.registry("tts")
    return {
        "status": "ready",
        "version": app.version,
        "device": device(),
        "services": ["transcription", "translation", "speech", "diarization", "pipeline"],
        "loaded": loaded(),
        # Legacy keys.
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
                "needs_reference": spec.reference_required,
            }
            for spec in tts.models()
        },
        # New summary; /capabilities has the detail.
        "asr_models": [spec.id for spec in asr.models() if spec.available()],
        "capabilities_url": "/capabilities",
        "languages": L.listing(),
    }


@app.get("/languages")
def languages() -> Dict:
    return {"languages": L.listing()}


# ==========================================================================
# Voice references
# ==========================================================================
def _reference_texts() -> Dict[str, str]:
    if not _REFERENCE_TEXT.exists():
        return {}
    try:
        return json.loads(_REFERENCE_TEXT.read_text(encoding="utf-8"))
    except ValueError:
        return {}


def _store_reference(upload: UploadFile, text: str) -> str:
    reference_id = jobs.new_id()[:16]
    target = _REFERENCE_DIR / f"{reference_id}.wav"
    with Scratch(Path(upload.filename or "ref.wav").suffix or ".wav") as raw:
        with raw.open("wb") as handle:
            shutil.copyfileobj(upload.file, handle)
        # Cloning models expect a clean mono clip; normalise whatever arrives.
        normalise_speech(raw, target)

    stored = sorted(_REFERENCE_DIR.glob("*.wav"), key=lambda path: path.stat().st_mtime)
    keep = {path.stem for path in stored[-_REFERENCE_LIMIT:]}
    for stale in stored[:-_REFERENCE_LIMIT]:
        stale.unlink(missing_ok=True)
    texts = {name: value for name, value in _reference_texts().items() if name in keep}
    if text.strip():
        texts[reference_id] = text.strip()
    _REFERENCE_TEXT.write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
    return reference_id


def _reference_path(reference_id: Optional[str]) -> Optional[Path]:
    if not reference_id:
        return None
    candidate = _REFERENCE_DIR / f"{Path(reference_id).name}.wav"
    return candidate if candidate.exists() else None


@app.post("/reference")
def upload_reference(
    audio: UploadFile = File(...),
    text: str = Form(""),
    authorization: Optional[str] = Header(None),
) -> Dict:
    """Store one speaker sample. `text` is what the clip says, which CosyVoice
    requires and F5 uses to skip transcribing the clip itself."""
    _authorize(authorization)
    reference_id = _store_reference(audio, text)
    return {"reference_id": reference_id, "has_text": bool(text.strip())}


# ==========================================================================
# Stage endpoints
# ==========================================================================
@app.post("/diarize")
def diarize(
    audio: UploadFile = File(...),
    provider: str = Form(""),
    model: str = Form(""),
    min_speakers: str = Form(""),
    max_speakers: str = Form(""),
    authorization: Optional[str] = Header(None),
) -> Dict:
    """Who spoke when, as turns the caller can merge with a transcript."""
    _authorize(authorization)
    spec = providers.find("diarization", provider or None,
                          model or providers.DEFAULTS["diarization"])
    providers.registry("diarization").require_available(spec)
    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    with Scratch(suffix) as path:
        with path.open("wb") as handle:
            shutil.copyfileobj(audio.file, handle)
        with jobs.lock:
            engine = providers.diarization.load(spec)
            turns = engine.diarize(
                path,
                min_speakers=int(min_speakers) if min_speakers.strip() else None,
                max_speakers=int(max_speakers) if max_speakers.strip() else None,
            )
    return {
        "turns": turns,
        "speakers": segment_tools.speakers_of(
            [{"speaker_id": turn["speaker_id"]} for turn in turns]
        ),
        "diarization_model": engine.name,
    }


@app.post("/transcribe")
def transcribe(
    audio: UploadFile = File(...),
    language: str = Form("auto"),
    model: str = Form(""),
    provider: str = Form(""),
    turns: str = Form(""),
    authorization: Optional[str] = Header(None),
) -> Dict:
    """Speech to text. `turns` is optional diarization JSON: when a recogniser
    has no timestamps of its own, those turns become its windows."""
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
    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    with Scratch(suffix) as path:
        with path.open("wb") as handle:
            shutil.copyfileobj(audio.file, handle)
        with jobs.lock:
            engine = providers.asr.load(spec, source)
            windows = (
                windows_from_turns(parsed)
                if parsed and not spec.supports_timestamps else None
            )
            # `source` may be None: a recogniser that detects the language does
            # so while transcribing, so asking it twice would run the model over
            # the whole file for nothing.
            result = engine.transcribe(path, source, windows)
        providers.registry("asr").require_language(spec, result.language, "source")

    segments = result.segments
    if parsed:
        segments = segment_tools.renumber(segment_tools.assign_speakers(segments, parsed))
    if not segments:
        raise ServiceError("No speech was found in the audio", status_code=422)
    return {
        "source_language": result.language,
        "segments": segments,
        "speakers": segment_tools.speakers_of(segments),
        "asr_provider": spec.provider,
        "asr_model": spec.id,
        # Legacy key: the old response called every recogniser "whisper".
        "whisper_model": spec.id,
    }


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
    #: Legacy name, still accepted.
    engine: str = ""
    provider: str = ""
    model: str = ""


@app.post("/translate")
def translate(
    request: TranslationRequest,
    authorization: Optional[str] = Header(None),
) -> Dict:
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

    with jobs.lock:
        engine = providers.translation.load(spec)
        translations = engine.translate(request.texts, source, target)
    if len(translations) != len(request.texts):
        raise ServiceError("The translation model returned the wrong number of rows")
    return {
        "translations": [text.strip() for text in translations],
        "translation_engine": spec.id,
        "translation_model": spec.id,
        "translation_provider": spec.provider,
        "skipped": False,
    }


@app.post("/synthesize")
def synthesize(
    text: str = Form(...),
    language: str = Form("vi"),
    speed: float = Form(1.0),
    engine: str = Form(""),
    provider: str = Form(""),
    model: str = Form(""),
    reference_id: str = Form(""),
    reference_text: str = Form(""),
    authorization: Optional[str] = Header(None),
) -> Response:
    _authorize(authorization)
    target = _language(language)
    spec = providers.find(
        "tts", provider or None, model or engine or providers.DEFAULTS["tts"]
    )
    providers.registry("tts").require_language(spec, target, "target")
    text = text.strip()
    if not text:
        raise InvalidRequest("Text is required")
    if len(text) > 2000:
        raise InvalidRequest("Text must be at most 2000 characters")
    if not 0.5 <= speed <= 2.0:
        raise InvalidRequest("Speed must be between 0.5 and 2.0")

    reference = _reference_path(reference_id)
    stored_text = _reference_texts().get(reference_id or "", "")
    with Scratch(".wav") as output:
        with jobs.lock:
            engine_instance = providers.tts.load(spec, target)
            engine_instance.synthesize(
                SpeechRequest(
                    text=text,
                    language=target,
                    speed=speed,
                    reference=reference,
                    reference_text=reference_text.strip() or stored_text,
                ),
                output,
            )
        payload = output.read_bytes()
        if not payload:
            raise ServiceError(f"The '{spec.id}' engine produced no audio")
        rate = sample_rate(output)
        length = duration(output)

    return Response(
        content=payload,
        media_type="audio/wav",
        headers={
            "X-TTS-Model": engine_instance.name,
            "X-TTS-Engine": spec.id,
            "X-TTS-Provider": spec.provider,
            "X-TTS-Sample-Rate": str(rate),
            "X-TTS-Duration": f"{length:.3f}",
        },
    )


@app.post("/separate")
def separate(
    audio: UploadFile = File(...),
    provider: str = Form(""),
    model: str = Form(""),
    stem: str = Form("background"),
    authorization: Optional[str] = Header(None),
) -> Response:
    """Split a track and return one stem.

    The caller asks for `background` (music and effects, the dub sits on this)
    or `speech` (the isolated original voice). Only the requested stem crosses
    the tunnel, which halves what is the largest transfer in the pipeline.
    """
    _authorize(authorization)
    if stem not in {"background", "speech"}:
        raise InvalidRequest("stem must be 'background' or 'speech'")
    spec = providers.find("separation", provider or None,
                          model or providers.DEFAULTS["separation"])
    providers.registry("separation").require_available(spec)
    workdir = Path(tempfile.mkdtemp(prefix="dubflow-separate-"))
    try:
        source = workdir / f"input{Path(audio.filename or 'audio.wav').suffix or '.wav'}"
        with source.open("wb") as handle:
            shutil.copyfileobj(audio.file, handle)
        with jobs.lock:
            engine = providers.separation.load(spec)
            stems = engine.separate(source, workdir)
        payload = stems[stem].read_bytes()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return Response(
        content=payload,
        media_type="audio/wav",
        headers={"X-Separation-Model": spec.id, "X-Separation-Stem": stem},
    )


class AlignmentRequest(BaseModel):
    segments: List[Dict[str, Any]]
    media_duration: Optional[float] = None
    min_speed: float = 0.0
    max_speed: float = 0.0
    allow_stretch: bool = False


@app.post("/align")
def align(request: AlignmentRequest, authorization: Optional[str] = Header(None)) -> Dict:
    """Plan the speed change for each segment without touching any audio.

    The caller applies the plan itself, which is how the local FFmpeg service
    aligns without shipping every clip to Colab and back.
    """
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


# ==========================================================================
# End-to-end pipeline
# ==========================================================================
@app.post("/validate")
def validate(request: Dict[str, Any], authorization: Optional[str] = Header(None)) -> Dict:
    """Check a job configuration without creating one.

    The local API calls this when a video is uploaded so an impossible
    combination is refused immediately rather than four stages later.
    """
    _authorize(authorization)
    return {"valid": True, "config": job_config.build(request).public()}


@app.post("/jobs", status_code=202)
async def create_job(
    request: Request,
    video: UploadFile = File(...),
    authorization: Optional[str] = Header(None),
) -> Dict:
    """Dub a video end to end. Returns immediately; poll GET /jobs/{id}.

    Every field except the video is read from the form, so the old three
    (`model`, `translation_engine`, `tts_engine`) and the new ones
    (`asr_provider`, `enable_diarization`, ...) are both accepted.
    """
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
    """The full canonical segments, including speakers and alignment metadata."""
    _authorize(authorization)
    job = jobs.read(job_id)
    return {
        "job_id": job_id,
        "segments": job.get("segments", []),
        "turns": job.get("turns", []),
        "references": job.get("references", {}),
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
    """Drop every resident model. Useful when a Colab GPU is nearly full."""
    _authorize(authorization)
    release_all()
    return {"loaded": loaded()}
