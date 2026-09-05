"""Local media API for the n8n dubbing workflow.

The service still has no database, no object storage and no models of its own:
it stores jobs, runs FFmpeg, and delegates every inference to a notebook
backend. What changed with the provider refactor is where the model lists live.
They are no longer duplicated here - `GET /capabilities` proxies the backend's
registry, and job configuration is validated by the same code that will run it.
The only tables kept locally are the ones in `dubflow_core`, which the Colab
service imports as well.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional

import httpx
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

APP_DIR = Path(__file__).resolve().parent
#: dubflow_core sits beside the service in the image and beside ai-service/ in
#: the repository. Both layouts resolve from here.
for candidate in (APP_DIR, APP_DIR.parent):
    if (candidate / "dubflow_core").is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from dubflow_core import alignment as align  # noqa: E402
from dubflow_core import languages as L  # noqa: E402
from dubflow_core import mixing  # noqa: E402
from dubflow_core import segments as segment_tools  # noqa: E402

ROOT = Path(os.getenv("DATA_ROOT", APP_DIR.parent / "data/jobs"))
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
#: The catalogue changes only when the notebook restarts.
CAPABILITIES_TTL = float(os.getenv("CAPABILITIES_TTL", "60"))
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL", "http://n8n:5678/webhook/dubbing/start")

FRONTEND_INDEX = Path(os.getenv("FRONTEND_INDEX", APP_DIR / "frontend" / "index.html"))
if not FRONTEND_INDEX.exists():
    # Local development keeps frontend/ beside ai-service/ at repository root.
    FRONTEND_INDEX = APP_DIR.parent / "frontend" / "index.html"

#: ISO-639-1 stays the public application code, from the shared table. Every
#: model-specific mapping belongs to the provider that needs it.
LANGUAGES = {code: row.name for code, row in L.LANGUAGES.items()}

_JOB_ID = re.compile(r"^[a-f0-9]{12}$")
_ALLOWED_VIDEO = {".mp4", ".mkv", ".mov", ".webm", ".m4v"}
#: Cloning quality plateaus well before this; longer clips only cost upload time.
_REFERENCE_SECONDS = 12.0
_REFERENCE_MINIMUM = 1.0
_MAX_REFERENCE_PIECE = 8.0

#: The stages n8n calls, in order. Optional ones skip themselves.
STAGES = [
    "extract",
    "diarize",
    "transcribe",
    "merge_segments",
    "translate",
    "voice_references",
    "synthesize",
    "align",
    "separate",
    "mix",
    "lipsync",
    "render",
]
FINAL_STAGE = "render"

app = FastAPI(title="Multilingual Dubbing API", version="3.0")


class JobRequest(BaseModel):
    job_id: str


def _job_dir(job_id: str) -> Path:
    if not _JOB_ID.fullmatch(job_id):
        raise HTTPException(400, "Invalid job_id")
    path = ROOT / job_id
    if not path.exists():
        raise HTTPException(404, f"Job '{job_id}' does not exist")
    return path


def _load_job(job_id: str) -> dict:
    return json.loads((_job_dir(job_id) / "job.json").read_text(encoding="utf-8"))


def _save_job(job: dict) -> None:
    path = _job_dir(job["job_id"]) / "job.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _features(job: dict) -> Dict[str, bool]:
    return (job.get("config") or {}).get("features") or {}


@contextmanager
def _stage(job_id: str, name: str) -> Iterator[tuple]:
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
        # Record it once: a retried stage that appended twice made the progress
        # count exceed the number of stages, and tripped the "already started"
        # guard on a job that had only run one node.
        completed = job.setdefault("completed_stages", [])
        if name not in completed:
            completed.append(name)
        if name in job.get("skipped_stages", []):
            job["skipped_stages"].remove(name)
        job["current_stage"] = None
        job["status"] = "completed" if name == FINAL_STAGE else "running"
        job.pop("error", None)
        _save_job(job)


def _skip(job_id: str, name: str, reason: str) -> dict:
    """Record an optional stage that decided not to run and move on."""
    job = _load_job(job_id)
    if name not in job.get("skipped_stages", []):
        job.setdefault("skipped_stages", []).append(name)
    job["current_stage"] = None
    _save_job(job)
    return {"job_id": job_id, "stage": name, "status": "skipped", "reason": reason}


def _run(command: List[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout or "command failed")[-3000:])


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


# ==========================================================================
# Notebook backends
# ==========================================================================
#: The backend that answered most recently, plus when it was checked.
_live_backend: Optional[tuple] = None
_live_checked = 0.0
_capabilities_cache: Optional[dict] = None
_capabilities_checked = 0.0


def _configured_backends() -> List[tuple]:
    """Return (name, url, token) for every backend with a usable URL."""
    backends = []
    for name in AI_BACKENDS:
        url = os.getenv(f"{name.upper()}_API_URL", "").strip().rstrip("/")
        if url.startswith(("http://", "https://")):
            backends.append((name, url, os.getenv(f"{name.upper()}_API_TOKEN", "")))
    return backends


def _misconfigured_backends() -> List[str]:
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
        response = httpx.get(f"{url}/health", headers=headers, timeout=BACKEND_PROBE_TIMEOUT)
    except httpx.RequestError:
        return False
    return response.status_code < 400


def _invalidate_backend() -> None:
    global _live_backend, _live_checked, _capabilities_cache, _capabilities_checked
    _live_backend = None
    _live_checked = 0.0
    _capabilities_cache = None
    _capabilities_checked = 0.0


def _resolve_backend(force: bool = False) -> Optional[tuple]:
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


def _colab_request(path: str, method: str = "POST", **kwargs):  # noqa: ANN003, ANN202
    """Call whichever notebook backend is alive. There is no local fallback."""
    backend = _resolve_backend()
    if backend is None:
        raise HTTPException(503, _backend_problem())

    # An upload stream cannot be replayed once a failed attempt has consumed
    # it, so only bodies we can rebuild are safe to retry elsewhere.
    repeatable = "files" not in kwargs
    attempted: List[str] = []
    while True:
        name, url, token = backend
        attempted.append(name)
        headers = dict(kwargs.get("headers") or {})
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            response = httpx.request(
                method,
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
                f"{_detail(response)}",
            )
        return response


def _detail(response) -> str:  # noqa: ANN001
    """The backend's own message, which now carries the useful part."""
    try:
        payload = response.json()
        if isinstance(payload, dict) and payload.get("detail"):
            return str(payload["detail"])[:800]
    except ValueError:
        pass
    return response.text[:500]


def _json(response) -> dict:  # noqa: ANN001
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError("The AI backend returned a non-JSON response") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("The AI backend returned an unexpected response shape")
    return payload


def _capabilities(force: bool = False) -> dict:
    """The live backend's registry, cached briefly."""
    global _capabilities_cache, _capabilities_checked
    now = time.monotonic()
    if not force and _capabilities_cache and now - _capabilities_checked < CAPABILITIES_TTL:
        return _capabilities_cache
    payload = _json(_colab_request("/capabilities", method="GET"))
    _capabilities_cache = payload
    _capabilities_checked = now
    return payload


# ==========================================================================
# Public routes
# ==========================================================================
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
        "version": app.version,
        "jobs_root": str(ROOT),
        "languages": len(LANGUAGES),
        "stages": STAGES,
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
    return {"languages": [{"code": code, "name": name} for code, name in LANGUAGES.items()]}


@app.get("/capabilities")
def capabilities() -> dict:
    """The backend's model registry, verbatim.

    The frontend reads this instead of holding its own copy of the model lists.
    When no notebook is running it answers with `available: false` and the same
    hint /backends gives, so the UI can explain itself rather than showing an
    empty form.
    """
    backend = _resolve_backend()
    if backend is None:
        return {
            "available": False,
            "reason": "offline",
            "hint": _backend_problem(),
            "languages": [{"code": code, "name": name} for code, name in LANGUAGES.items()],
            "providers": {},
            "stages": STAGES,
        }
    try:
        payload = dict(_capabilities())
    except HTTPException as exc:
        detail = str(exc.detail)
        reason = "error"
        if "404" in detail:
            reason = "outdated"
            # The session answers /health but has no registry: it is running a
            # build from before the provider refactor. Say so, because "no
            # backend" is the one thing this is not.
            detail = (
                f"The '{backend[0]}' session at {backend[1]} is alive but has no "
                f"/capabilities endpoint, so it is running an older build of the "
                f"notebook. Re-run all cells in colab/ai_service.ipynb to pick up "
                f"the model registry, then point .env at the new tunnel URL."
            )
        return {
            "available": False,
            "reason": reason,
            "hint": detail,
            "languages": [{"code": code, "name": name} for code, name in LANGUAGES.items()],
            "providers": {},
            "stages": STAGES,
        }
    payload["available"] = True
    payload["ai_backend"] = backend[0]
    payload.setdefault("stages", STAGES)
    return payload


@app.post("/jobs/upload")
async def upload_video(request: Request, file: UploadFile = File(...)) -> dict:
    """Store a video and settle its configuration.

    Every non-file form field is forwarded to the backend's validator, so the
    old three parameters and the new ones are accepted without this service
    keeping a second copy of the rules. An impossible combination - a Vietnamese
    dub with an engine that has no Vietnamese - is refused here rather than
    four stages later.
    """
    form = await request.form()
    values = {
        name: value for name, value in form.items()
        if name != "file" and isinstance(value, str)
    }
    values.setdefault("target_language", "vi")
    values.setdefault("source_language", "auto")

    suffix = Path(file.filename or "input.mp4").suffix.lower() or ".mp4"
    if suffix not in _ALLOWED_VIDEO:
        raise HTTPException(400, f"Unsupported video container '{suffix}'")

    config = _json(_colab_request("/validate", json=values)).get("config") or {}
    if not config:
        raise HTTPException(502, "The AI backend returned no configuration")

    job_id = uuid.uuid4().hex[:12]
    folder = ROOT / job_id
    folder.mkdir(parents=True)
    source_path = folder / f"input{suffix}"
    with source_path.open("wb") as output:
        shutil.copyfileobj(file.file, output, length=8 * 1024 * 1024)

    source = config.get("source_language")
    job = {
        "job_id": job_id,
        "status": "running",
        "current_stage": None,
        "completed_stages": ["upload"],
        "skipped_stages": [],
        "source_filename": file.filename,
        "config": config,
        "request": values,
        "source_language": None if source in {None, "auto"} else source,
        "target_language": config.get("target_language"),
        # Legacy keys, so an existing dashboard keeps working.
        "whisper_model": (config.get("asr") or {}).get("model"),
        "translation_engine": (config.get("translation") or {}).get("model"),
        "tts_engine": (config.get("tts") or {}).get("model"),
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
        "config": config,
        "target_language": job["target_language"],
        "whisper_model": job["whisper_model"],
        "translation_engine": job["translation_engine"],
        "tts_engine": job["tts_engine"],
    }


# ==========================================================================
# Stages
# ==========================================================================
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


@app.post("/stages/diarize")
def diarize(request: JobRequest) -> dict:
    """Who spoke when. Skipped unless the job asked for it."""
    job = _load_job(request.job_id)
    if not _features(job).get("diarization"):
        return _skip(request.job_id, "diarize", "diarization is off for this job")

    with _stage(request.job_id, "diarize") as (job, folder):
        audio = folder / job["files"]["asr_audio"]
        choice = job["config"].get("diarization") or {}
        with audio.open("rb") as stream:
            payload = _json(_colab_request(
                "/diarize",
                files={"audio": (audio.name, stream, "audio/wav")},
                data={
                    "provider": choice.get("provider", ""),
                    "model": choice.get("model", ""),
                    "min_speakers": job["request"].get("min_speakers", ""),
                    "max_speakers": job["request"].get("max_speakers", ""),
                },
            ))
        turns = payload.get("turns") or []
        if not turns:
            raise RuntimeError("Diarization returned no speaker turns")
        job["turns"] = turns
        job["speakers"] = payload.get("speakers") or []
        job["diarization_model"] = payload.get("diarization_model")
        (folder / "diarization.json").write_text(
            json.dumps(turns, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        job["files"]["diarization"] = "diarization.json"
    return {
        "job_id": request.job_id,
        "stage": "diarize",
        "status": "completed",
        "speakers": job["speakers"],
        "turns": len(job["turns"]),
    }


@app.post("/stages/transcribe")
def transcribe(request: JobRequest) -> dict:
    with _stage(request.job_id, "transcribe") as (job, folder):
        audio = folder / job["files"]["asr_audio"]
        choice = job["config"].get("asr") or {}
        with audio.open("rb") as stream:
            payload = _json(_colab_request(
                "/transcribe",
                files={"audio": (audio.name, stream, "audio/wav")},
                data={
                    # An empty language means auto-detect. Anything else goes
                    # straight to the recogniser, which rejects the literal
                    # "auto" as an invalid language code.
                    "language": job.get("source_language") or "",
                    "provider": choice.get("provider", ""),
                    "model": choice.get("model", ""),
                    "turns": json.dumps(job.get("turns") or []),
                },
            ))
        segments = payload.get("segments") or []
        detected = payload.get("source_language")
        if not segments:
            raise RuntimeError("The recogniser found no speech in the video")
        if detected not in LANGUAGES:
            raise RuntimeError(f"Detected language '{detected}' is not configured")
        job["source_language"] = detected
        job["segments"] = segments
        job["asr_model"] = payload.get("asr_model")
        (folder / "transcript.json").write_text(
            json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return {
        "job_id": request.job_id,
        "stage": "transcribe",
        "status": "completed",
        "source_language": job["source_language"],
        "asr_model": job.get("asr_model"),
        # Legacy key.
        "whisper_model": job.get("asr_model"),
        "segment_count": len(job["segments"]),
    }


@app.post("/stages/merge_segments")
def merge_segments(request: JobRequest) -> dict:
    """Give every segment a speaker. Pure local logic, shared with Colab."""
    with _stage(request.job_id, "merge_segments") as (job, folder):
        segments = segment_tools.renumber(
            segment_tools.assign_speakers(job.get("segments", []), job.get("turns") or [])
        )
        job["segments"] = segments
        job["speakers"] = segment_tools.speakers_of(segments)
        job["speaker_summary"] = segment_tools.public_speaker_summary(segments)
        (folder / "transcript.json").write_text(
            json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return {
        "job_id": request.job_id,
        "stage": "merge_segments",
        "status": "completed",
        "speakers": job["speakers"],
        "segment_count": len(job["segments"]),
    }


@app.post("/stages/translate")
def translate(request: JobRequest) -> dict:
    with _stage(request.job_id, "translate") as (job, folder):
        source = job["source_language"]
        target = job["config"]["target_language"]
        choice = job["config"].get("translation") or {}
        texts = [segment["source_text"] for segment in job["segments"]]

        if source == target:
            translations = texts
        else:
            payload = _json(_colab_request(
                "/translate",
                json={
                    "source_language": source,
                    "target_language": target,
                    "texts": texts,
                    "provider": choice.get("provider", ""),
                    "model": choice.get("model", ""),
                },
            ))
            translations = payload.get("translations")
            if not isinstance(translations, list):
                raise RuntimeError("The AI backend returned no translations")
            if len(translations) != len(texts):
                raise RuntimeError("The AI backend returned the wrong number of translations")

        for segment, translated in zip(job["segments"], translations):
            segment["translated_text"] = (translated or "").strip()
        subtitle = folder / "translated.srt"
        subtitle.write_text(segment_tools.subtitle(job["segments"]), encoding="utf-8")
        job["files"]["subtitle"] = subtitle.name
    return {
        "job_id": request.job_id,
        "stage": "translate",
        "status": "completed",
        "source_language": source,
        "target_language": target,
        "translation_engine": choice.get("model"),
        "skipped_translation": source == target,
        "segment_count": len(job["segments"]),
    }


def _reference_regions(job: dict, speaker: str) -> List[dict]:
    """Where this speaker talks alone, longest first.

    With diarization these are the parts of the speaker's turns that nobody
    overlaps. Without it there is one speaker, so their own segments are used -
    which is what the single-reference version did.
    """
    turns = job.get("turns") or []
    if turns:
        regions = segment_tools.exclusive_regions(
            turns, speaker, min_duration=_REFERENCE_MINIMUM
        )
        if regions:
            return regions
    return [
        {"start": float(item["start"]), "end": float(item["end"]),
         "duration": float(item["end"]) - float(item["start"])}
        for item in sorted(
            (segment for segment in job["segments"]
             if (segment.get("speaker_id") or segment_tools.DEFAULT_SPEAKER) == speaker),
            key=lambda segment: float(segment["end"]) - float(segment["start"]),
            reverse=True,
        )
    ]


@app.post("/stages/voice_references")
def voice_references(request: JobRequest) -> dict:
    """Cut and upload one clean reference clip per speaker.

    The engines that clone a voice get one sample per speaker instead of a
    single clip from the top of the video, so a two-person conversation keeps
    two voices.
    """
    job = _load_job(request.job_id)
    if not _features(job).get("voice_cloning"):
        return _skip(request.job_id, "voice_references", "voice cloning is off for this job")

    with _stage(request.job_id, "voice_references") as (job, folder):
        if not job.get("segments"):
            raise RuntimeError("A voice reference needs the transcript, which is empty")
        original = folder / job["files"]["original_audio"]
        output_dir = folder / "references"
        output_dir.mkdir(exist_ok=True)

        references: Dict[str, dict] = {}
        for speaker in job.get("speakers") or [segment_tools.DEFAULT_SPEAKER]:
            chosen, total = [], 0.0
            for region in _reference_regions(job, speaker):
                if total >= _REFERENCE_SECONDS:
                    break
                length = min(
                    region["duration"], _MAX_REFERENCE_PIECE, _REFERENCE_SECONDS - total
                )
                if length < 0.4:
                    continue
                chosen.append({"start": region["start"], "length": round(length, 3)})
                total += length
            if total < _REFERENCE_MINIMUM:
                raise RuntimeError(
                    f"{speaker} has only {total:.2f}s of clean speech, too little to "
                    f"clone a voice. Turn voice cloning off, or use a clip where "
                    f"each speaker talks for at least a second."
                )

            target = output_dir / f"{speaker}.wav"
            pieces = []
            for index, piece in enumerate(chosen):
                part = output_dir / f".{speaker}.{index}.wav"
                _run([
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-ss", f"{piece['start']:.3f}", "-t", f"{piece['length']:.3f}",
                    "-i", str(original), "-vn", "-ar", "24000", "-ac", "1",
                    "-c:a", "pcm_s16le", str(part),
                ])
                pieces.append(part)
            if len(pieces) == 1:
                pieces[0].replace(target)
            else:
                inputs = []
                for part in pieces:
                    inputs.extend(["-i", str(part)])
                labels = "".join(f"[{index}:a]" for index in range(len(pieces)))
                _run([
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *inputs,
                    "-filter_complex",
                    f"{labels}concat=n={len(pieces)}:v=0:a=1,aresample=24000[out]",
                    "-map", "[out]", "-c:a", "pcm_s16le", str(target),
                ])
                for part in pieces:
                    part.unlink(missing_ok=True)

            spoken = " ".join(
                segment_tools.text_between(
                    job["segments"], piece["start"], piece["start"] + piece["length"]
                )
                for piece in chosen
            ).strip()
            with target.open("rb") as stream:
                payload = _json(_colab_request(
                    "/reference",
                    files={"audio": (target.name, stream, "audio/wav")},
                    data={"text": spoken},
                ))
            reference_id = payload.get("reference_id")
            if not reference_id:
                raise RuntimeError("The AI backend returned no reference_id")
            references[speaker] = {
                "file": str(target.relative_to(folder)),
                "reference_id": reference_id,
                "seconds": round(total, 3),
                "text": spoken,
                "exclusive": bool(job.get("turns")),
            }

        job["references"] = references
        job["files"]["references"] = "references"
    return {
        "job_id": request.job_id,
        "stage": "voice_references",
        "status": "completed",
        "speakers": list(job["references"]),
        "seconds": {name: entry["seconds"] for name, entry in job["references"].items()},
    }


@app.post("/stages/synthesize")
def synthesize(request: JobRequest) -> dict:
    with _stage(request.job_id, "synthesize") as (job, folder):
        target = job["config"]["target_language"]
        choice = job["config"].get("tts") or {}
        references = job.get("references") or {}
        output_dir = folder / "tts"
        output_dir.mkdir(exist_ok=True)

        models: Dict[str, int] = {}
        for segment in job["segments"]:
            text = (segment.get("translated_text") or "").strip()
            if not text:
                continue
            reference = references.get(segment.get("speaker_id")) or {}
            data = {
                "text": text,
                "language": target,
                "speed": "1.0",
                "provider": choice.get("provider", ""),
                "model": choice.get("model", ""),
            }
            if reference.get("reference_id"):
                data["reference_id"] = reference["reference_id"]
            if reference.get("text"):
                data["reference_text"] = reference["text"]
            response = _colab_request("/synthesize", data=data)
            output = output_dir / f"{segment['id']:04d}.wav"
            output.write_bytes(response.content)
            if not output.exists() or output.stat().st_size == 0:
                raise RuntimeError("The AI backend returned an empty audio file")
            model_name = response.headers.get("x-tts-model") or choice.get("model")
            segment["tts_file"] = str(output.relative_to(folder))
            segment["tts_duration_raw"] = round(_duration(output), 3)
            segment["tts_duration"] = segment["tts_duration_raw"]
            segment["tts_model"] = model_name
            models[model_name] = models.get(model_name, 0) + 1
        job["tts_provider"] = "colab"
        job["models_used"] = models

    generated = sum(1 for segment in job["segments"] if segment.get("tts_file"))
    return {
        "job_id": request.job_id,
        "stage": "synthesize",
        "status": "completed",
        "generated_segments": generated,
        "requested_engine": (job["config"].get("tts") or {}).get("model"),
        "voice_reference": bool(job.get("references")),
        "speakers": list(job.get("references") or {}),
        "provider": job["tts_provider"],
        "models_used": job.get("models_used"),
    }


@app.post("/stages/align")
def align_speech(request: JobRequest) -> dict:
    """Fit each generated line into the gap it has to fill.

    The arithmetic is `dubflow_core.alignment`, the same module the Colab
    pipeline uses, so no audio has to cross the tunnel for this.
    """
    job = _load_job(request.job_id)
    if not _features(job).get("alignment"):
        return _skip(request.job_id, "align", "alignment is off for this job")

    with _stage(request.job_id, "align") as (job, folder):
        limits_config = job["config"].get("alignment_limits") or {}
        limits = align.Limits(
            min_speed=float(limits_config.get("min_speed", align.DEFAULT_MIN_SPEED)),
            max_speed=float(limits_config.get("max_speed", align.DEFAULT_MAX_SPEED)),
            tolerance=float(limits_config.get("tolerance", align.DEFAULT_TOLERANCE)),
            allow_stretch=bool(limits_config.get("allow_stretch", False)),
        )
        plans = align.plan_segments(job["segments"], job.get("duration_seconds"), limits)
        for segment, plan in zip(job["segments"], plans):
            path = segment.get("tts_file")
            if not path:
                align.record(segment, align.Plan(1.0, "unmeasured", plan.window, 0.0), None)
                continue
            clip = folder / path
            if abs(plan.speed - 1.0) <= 1e-3:
                align.record(segment, plan, segment.get("tts_duration_raw"))
                continue
            retimed = clip.with_suffix(".aligned.wav")
            _run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(clip), "-af", mixing.atempo_filter(plan.speed),
                "-c:a", "pcm_s16le", str(retimed),
            ])
            retimed.replace(clip)
            align.record(segment, plan, _duration(clip))
        job["alignment"] = align.summary(job["segments"])
    return {
        "job_id": request.job_id,
        "stage": "align",
        "status": "completed",
        "alignment": job["alignment"],
    }


@app.post("/stages/separate")
def separate(request: JobRequest) -> dict:
    """Replace the original track with its background stem. Optional."""
    job = _load_job(request.job_id)
    if not _features(job).get("source_separation"):
        return _skip(request.job_id, "separate", "source separation is off for this job")

    with _stage(request.job_id, "separate") as (job, folder):
        original = folder / job["files"]["original_audio"]
        choice = job["config"].get("separation") or {}
        with original.open("rb") as stream:
            response = _colab_request(
                "/separate",
                files={"audio": (original.name, stream, "audio/wav")},
                data={
                    "provider": choice.get("provider", ""),
                    "model": choice.get("model", ""),
                    "stem": "background",
                },
            )
        background = folder / "background.wav"
        background.write_bytes(response.content)
        if background.stat().st_size == 0:
            raise RuntimeError("The AI backend returned an empty background stem")
        job["files"]["background_audio"] = background.name
        job["separation_model"] = response.headers.get("x-separation-model")
    return {
        "job_id": request.job_id,
        "stage": "separate",
        "status": "completed",
        "model": job.get("separation_model"),
    }


@app.post("/stages/mix")
def mix(request: JobRequest) -> dict:
    with _stage(request.job_id, "mix") as (job, folder):
        voiced = segment_tools.voiced(job["segments"])
        if not voiced:
            raise RuntimeError("No generated speech is available")

        dubbed = folder / "dubbed.wav"
        command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
        for segment in voiced:
            command.extend(["-i", str(folder / segment["tts_file"])])
        duration = float(job["duration_seconds"])
        command.extend([
            "-filter_complex",
            mixing.dub_filtergraph([segment["start"] for segment in voiced], duration),
            "-map", "[dub]", "-ar", str(mixing.MIX_SAMPLE_RATE), "-ac", "2",
            "-c:a", "pcm_s16le", str(dubbed),
        ])
        _run(command)

        # Without separation the original soundtrack is kept at low volume,
        # producing a voice-over. With it the speech stem is already gone, so
        # the background keeps its level and the dub sits on top of it. Both
        # graphs live in dubflow_core so the two routes cannot drift apart.
        background_name = job["files"].get("background_audio")
        separated = bool(background_name)
        background = folder / (background_name or job["files"]["original_audio"])

        final_audio = folder / "final.wav"
        _run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(background), "-i", str(dubbed),
            "-filter_complex", mixing.blend_filtergraph(separated),
            "-map", "[out]", "-ar", str(mixing.MIX_SAMPLE_RATE), "-ac", "2",
            "-c:a", "pcm_s16le", str(final_audio),
        ])
        job["files"].update({"dubbed_audio": dubbed.name, "final_audio": final_audio.name})
        job["mix_mode"] = mixing.mix_mode(separated)
    return {
        "job_id": request.job_id,
        "stage": "mix",
        "status": "completed",
        "mode": job["mix_mode"],
        "placed_segments": len(voiced),
    }


@app.post("/stages/lipsync")
def lipsync(request: JobRequest) -> dict:
    """Optional lip sync.

    This route has no implementation on purpose. Lip sync rewrites the video,
    and sending the whole file to the notebook and back would dwarf every other
    transfer in this pipeline - the point of the n8n route is that only a 16 kHz
    track crosses the tunnel. A job that wants lip sync should use the
    notebook's own end-to-end `POST /jobs`, where the video is already there.
    """
    job = _load_job(request.job_id)
    if not _features(job).get("lip_sync"):
        return _skip(request.job_id, "lipsync", "lip sync is off for this job")
    raise HTTPException(
        501,
        "This route does not run lip sync: the video would have to cross the "
        "tunnel twice. Use the AI service's own POST /jobs for a lip-synced dub, "
        "and note that no lip-sync provider ships with this build either - see "
        "colab/providers/lipsync/registry.py.",
    )


@app.post("/stages/render")
def render(request: JobRequest) -> dict:
    with _stage(request.job_id, "render") as (job, folder):
        files = job["files"]
        video = folder / (files.get("lipsync_video") or files["input"])
        audio = folder / files["final_audio"]
        subtitle = folder / files["subtitle"]
        target = job["config"]["target_language"]
        output = folder / f"dubbed_{target}.mp4"
        duration = float(job["duration_seconds"])
        base = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(video), "-i", str(audio), "-i", str(subtitle),
            "-map", "0:v:0", "-map", "1:a:0", "-map", "2:0",
        ]
        finish = [
            "-c:a", "aac", "-b:a", "192k", "-c:s", "mov_text",
            "-metadata:s:s:0", f"language={target}",
            "-t", f"{duration:.3f}", "-movflags", "+faststart", str(output),
        ]
        try:
            _run(base + ["-c:v", "copy"] + finish)
        except RuntimeError:
            _run(base + ["-c:v", "libx264", "-preset", "fast", "-crf", "22"] + finish)
        job["files"]["output"] = output.name
        base_url = os.getenv("PUBLIC_BASE_URL", "http://localhost:8000")
        job["output_url"] = f"{base_url}/jobs/{request.job_id}/download"
    return {
        "job_id": request.job_id,
        "stage": "render",
        "status": "completed",
        "output_url": job["output_url"],
        "subtitle_url": f"{os.getenv('PUBLIC_BASE_URL', 'http://localhost:8000')}"
                        f"/jobs/{request.job_id}/subtitle",
    }


# ==========================================================================
# Job access
# ==========================================================================
@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    return _load_job(job_id)


@app.post("/jobs/{job_id}/start", status_code=202)
def start_job(job_id: str) -> dict:
    """Ask n8n to orchestrate the stages after upload."""
    job = _load_job(job_id)
    completed = job.get("completed_stages", [])
    if job.get("status") == "completed" or len(completed) > 1:
        return {
            "job_id": job_id,
            "status": job.get("status", "running"),
            "message": "Job has already started",
        }

    try:
        response = httpx.post(N8N_WEBHOOK_URL, json={"job_id": job_id}, timeout=30)
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
