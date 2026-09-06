"""Local media API for the n8n dubbing workflow."""
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
from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional

import httpx
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

APP_DIR = Path(__file__).resolve().parent
for candidate in (APP_DIR, APP_DIR.parent):
    if (candidate / "dubflow_core").is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from dubflow_core import alignment as align  # noqa: E402
from dubflow_core import languages as L  # noqa: E402
from dubflow_core import mixing  # noqa: E402
from dubflow_core import segments as segment_tools  # noqa: E402

ROOT = Path(os.getenv("DATA_ROOT", APP_DIR.parent / "data/jobs"))
ROOT.mkdir(parents=True, exist_ok=True)

COLAB_API_TIMEOUT = float(os.getenv("COLAB_API_TIMEOUT", "1800"))
AI_BACKENDS = [
    name.strip().lower()
    for name in os.getenv("AI_BACKENDS", "colab,kaggle").split(",")
    if name.strip()
]
BACKEND_PROBE_TTL = float(os.getenv("BACKEND_PROBE_TTL", "30"))
BACKEND_PROBE_TIMEOUT = float(os.getenv("BACKEND_PROBE_TIMEOUT", "10"))
CAPABILITIES_TTL = float(os.getenv("CAPABILITIES_TTL", "60"))
# Avoid holding one request open through long model stages or tunnel timeouts.
ASYNC_STAGES = os.getenv("ASYNC_STAGES", "1").strip().lower() not in {"0", "false", "no"}
TASK_POLL_INTERVAL = float(os.getenv("TASK_POLL_INTERVAL", "3"))
TASK_POLL_TIMEOUT = float(os.getenv("TASK_POLL_TIMEOUT", "3600"))
# n8n cannot mark a job failed if it restarts while waiting for a stage.
STALLED_AFTER = float(os.getenv("STALLED_AFTER", "1200"))
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL", "http://n8n:5678/webhook/dubbing/start")

FRONTEND_INDEX = Path(os.getenv("FRONTEND_INDEX", APP_DIR / "frontend" / "index.html"))
if not FRONTEND_INDEX.exists():
    FRONTEND_INDEX = APP_DIR.parent / "frontend" / "index.html"

LANGUAGES = {code: row.name for code, row in L.LANGUAGES.items()}

_JOB_ID = re.compile(r"^[a-f0-9]{12}$")
_ALLOWED_VIDEO = {".mp4", ".mkv", ".mov", ".webm", ".m4v"}

MODERN_ENDPOINTS = ("/capabilities", "/validate", "/diarize", "/separate", "/align")
MIN_BACKEND_VERSION = "5.0"

STAGES = [
    "extract",
    "diarize",
    "transcribe",
    "merge_segments",
    "translate",
    "synthesize",
    "align",
    "separate",
    "mix",
    "render",
]
FINAL_STAGE = "render"

app = FastAPI(title="Multilingual Dubbing API", version="3.0")


class OutdatedBackend(HTTPException):
    def __init__(self, detail: str) -> None:
        super().__init__(502, detail)


class JobRequest(BaseModel):
    job_id: str
    force: bool = False


def _job_dir(job_id: str) -> Path:
    if not _JOB_ID.fullmatch(job_id):
        raise HTTPException(400, "Invalid job_id")
    path = ROOT / job_id
    if not path.exists():
        raise HTTPException(404, f"Job '{job_id}' does not exist")
    return path


def _removable_job_dir(job_id: str) -> Path:
    """Resolve only direct children of the job root for deletion."""
    candidate = (ROOT / job_id).resolve()
    if candidate.parent != ROOT.resolve() or not candidate.is_dir():
        raise HTTPException(404, f"Job '{job_id}' does not exist")
    return candidate


def _load_job(job_id: str) -> dict:
    return json.loads((_job_dir(job_id) / "job.json").read_text(encoding="utf-8"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


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
    job["stage_started_at"] = _now()
    _save_job(job)
    try:
        yield job, folder
    except Exception as exc:
        message = exc.detail if isinstance(exc, HTTPException) else str(exc)
        job["status"] = "failed"
        job["error"] = {"stage": name, "message": str(message)[:1000]}
        job["finished_at"] = _now()
        _save_job(job)
        if isinstance(exc, HTTPException):
            raise
        # Preserve the real error for n8n instead of returning an empty 500.
        raise HTTPException(500, f"Stage '{name}' failed: {message}"[:1000]) from exc
    else:
        completed = job.setdefault("completed_stages", [])
        if name not in completed:
            completed.append(name)
        if name in job.get("skipped_stages", []):
            job["skipped_stages"].remove(name)
        job["current_stage"] = None
        job["status"] = "completed" if name == FINAL_STAGE else "running"
        if name == FINAL_STAGE:
            job["finished_at"] = _now()
        job.pop("error", None)
        _save_job(job)


def _reuse(job_id: str, name: str, force: bool) -> Optional[dict]:
    """Reuse a completed stage unless the caller forces a rerun."""
    job = _load_job(job_id)
    if force or name not in job.get("completed_stages", []):
        return None
    return {
        "job_id": job_id,
        "stage": name,
        "status": "completed",
        "reused": True,
        "reason": "this stage already produced its output",
    }


def _skip(job_id: str, name: str, reason: str) -> dict:
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


_live_backend: Optional[tuple] = None
_live_checked = 0.0
_capabilities_cache: Optional[dict] = None
_capabilities_checked = 0.0


def _configured_backends() -> List[tuple]:
    backends = []
    for name in AI_BACKENDS:
        url = os.getenv(f"{name.upper()}_API_URL", "").strip().rstrip("/")
        if url.startswith(("http://", "https://")):
            backends.append((name, url, os.getenv(f"{name.upper()}_API_TOKEN", "")))
    return backends


def _misconfigured_backends() -> List[str]:
    broken = []
    for name in AI_BACKENDS:
        url = os.getenv(f"{name.upper()}_API_URL", "").strip()
        if url and not url.startswith(("http://", "https://")):
            broken.append(name)
    return broken


def _backend_version(url: str, token: str) -> Optional[str]:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        payload = httpx.get(
            f"{url}/health", headers=headers, timeout=BACKEND_PROBE_TIMEOUT
        ).json()
    except (httpx.RequestError, ValueError):
        return None
    version = payload.get("version") if isinstance(payload, dict) else None
    return str(version) if version else None


def _outdated_message(name: str, url: str, token: str, path: str) -> str:
    """Explain how to replace an outdated backend process."""
    seen = _backend_version(url, token)
    if name == "local":
        action = "Rebuild it with: make local-restart"
    else:
        action = (
            "In Colab: Runtime > Restart session, then Run all - re-running the "
            "cells without a restart keeps the old code loaded. Then point this "
            "stack at the new tunnel with: make colab URL=<url> TOKEN=<token>"
        )
    return (
        f"The '{name}' session at {url} is running AI service "
        f"{seen or 'a build older than ' + MIN_BACKEND_VERSION}, which has no "
        f"{path} endpoint. This stack needs {MIN_BACKEND_VERSION} or newer. "
        f"{action}"
    )


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
    actions = []
    if any(name == "local" for name, _, _ in configured):
        actions.append("restart the local backend with 'make local-restart'")
    if any(name != "local" for name, _, _ in configured):
        actions.append("restart the notebook and update its URL in .env")
    return f"No configured AI backend answered /health: {listed}. " + " or ".join(actions) + "."


def _colab_request(path: str, method: str = "POST", **kwargs):  # noqa: ANN003, ANN202
    """Call whichever configured AI backend is alive."""
    backend = _resolve_backend()
    if backend is None:
        raise HTTPException(503, _backend_problem())

    # Upload streams cannot be replayed safely after a failed attempt.
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
        if response.status_code == 404 and path in MODERN_ENDPOINTS:
            raise OutdatedBackend(_outdated_message(name, url, token, path))
        if response.status_code >= 400:
            raise HTTPException(
                502,
                f"AI backend '{name}' returned {response.status_code}: "
                f"{_detail(response)}",
            )
        return response


def _detail(response) -> str:  # noqa: ANN001
    try:
        payload = response.json()
        if isinstance(payload, dict) and payload.get("detail"):
            return str(payload["detail"])[:800]
    except ValueError:
        pass
    return response.text[:500]


def _await_task(state: dict, path: str) -> dict:
    """Poll an asynchronous backend task until it finishes."""
    task_id = state.get("task_id")
    if not task_id:
        raise RuntimeError(f"The AI backend accepted {path} but returned no task id")
    deadline = time.monotonic() + TASK_POLL_TIMEOUT
    while True:
        if time.monotonic() > deadline:
            raise HTTPException(
                504,
                f"The AI backend is still running {path} after "
                f"{int(TASK_POLL_TIMEOUT / 60)} minutes (task {task_id}). It may "
                f"still finish; raise TASK_POLL_TIMEOUT if this video is simply long.",
            )
        time.sleep(TASK_POLL_INTERVAL)
        payload = _json(_colab_request(f"/tasks/{task_id}", method="GET"))
        if not payload.get("done"):
            continue
        if payload.get("status") == "failed":
            raise HTTPException(
                502,
                f"AI backend failed during {path}: "
                f"{payload.get('error') or 'no reason given'}",
            )
        return payload.get("result") or {}


def _stage_call(path: str, **kwargs) -> dict:  # noqa: ANN003
    """Run a stage asynchronously when the backend supports it."""
    if not ASYNC_STAGES:
        return _json(_colab_request(path, **kwargs))
    payload = dict(kwargs)
    if "json" in payload:
        payload["json"] = {**payload["json"], "async_mode": True}
    else:
        payload["data"] = {**(payload.get("data") or {}), "async_mode": "true"}
    response = _colab_request(path, **payload)
    if response.status_code != 202:
        return _json(response)
    return _await_task(_json(response), path)


def _json(response) -> dict:  # noqa: ANN001
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError("The AI backend returned a non-JSON response") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("The AI backend returned an unexpected response shape")
    return payload


def _capabilities(force: bool = False) -> dict:
    global _capabilities_cache, _capabilities_checked
    now = time.monotonic()
    if not force and _capabilities_cache and now - _capabilities_checked < CAPABILITIES_TTL:
        return _capabilities_cache
    payload = _json(_colab_request("/capabilities", method="GET"))
    _capabilities_cache = payload
    _capabilities_checked = now
    return payload


@app.get("/", include_in_schema=False)
def demo_frontend():  # noqa: ANN201
    if not FRONTEND_INDEX.exists():
        raise HTTPException(404, "Demo frontend is not installed")
    return FileResponse(FRONTEND_INDEX, media_type="text/html")


@app.get("/health")
def health() -> dict:
    backend = _resolve_backend()
    return {
        "status": "ready",
        "version": app.version,
        "jobs_root": str(ROOT),
        "languages": len(LANGUAGES),
        "stages": STAGES,
        "ai_backend": backend[0] if backend else None,
        "ai_backend_url": backend[1] if backend else None,
    }


@app.get("/backends")
def backends() -> dict:
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
    """Return the active backend's model registry."""
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
        return {
            "available": False,
            "reason": "outdated" if isinstance(exc, OutdatedBackend) else "error",
            "hint": str(exc.detail),
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
    """Store a video after the backend validates its configuration."""
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
        "created_at": _now(),
        "finished_at": None,
        "source_filename": file.filename,
        "config": config,
        "request": values,
        "source_language": None if source in {None, "auto"} else source,
        "target_language": config.get("target_language"),
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
    }


@app.post("/stages/extract")
def extract_audio(request: JobRequest) -> dict:
    reused = _reuse(request.job_id, "extract", request.force)
    if reused:
        return reused
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
    reused = _reuse(request.job_id, "diarize", request.force)
    if reused:
        return reused
    job = _load_job(request.job_id)
    with _stage(request.job_id, "diarize") as (job, folder):
        audio = folder / job["files"]["asr_audio"]
        choice = job["config"].get("diarization") or {}
        with audio.open("rb") as stream:
            payload = _stage_call(
                "/diarize",
                files={"audio": (audio.name, stream, "audio/wav")},
                data={
                    "provider": choice.get("provider", ""),
                    "model": choice.get("model", ""),
                    "min_speakers": job["request"].get("min_speakers", ""),
                    "max_speakers": job["request"].get("max_speakers", ""),
                },
            )
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
    reused = _reuse(request.job_id, "transcribe", request.force)
    if reused:
        return reused
    with _stage(request.job_id, "transcribe") as (job, folder):
        audio = folder / job["files"]["asr_audio"]
        choice = job["config"].get("asr") or {}
        with audio.open("rb") as stream:
            payload = _stage_call(
                "/transcribe",
                files={"audio": (audio.name, stream, "audio/wav")},
                data={
                    # Empty means auto-detect; "auto" is not a model language code.
                    "language": job.get("source_language") or "",
                    "provider": choice.get("provider", ""),
                    "model": choice.get("model", ""),
                    "turns": json.dumps(job.get("turns") or []),
                },
            )
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
        "segment_count": len(job["segments"]),
    }


@app.post("/stages/merge_segments")
def merge_segments(request: JobRequest) -> dict:
    reused = _reuse(request.job_id, "merge_segments", request.force)
    if reused:
        return reused
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
    reused = _reuse(request.job_id, "translate", request.force)
    if reused:
        return reused
    with _stage(request.job_id, "translate") as (job, folder):
        source = job["source_language"]
        target = job["config"]["target_language"]
        choice = job["config"].get("translation") or {}
        texts = [segment["source_text"] for segment in job["segments"]]

        if source == target:
            translations = texts
        else:
            payload = _stage_call(
                "/translate",
                json={
                    "source_language": source,
                    "target_language": target,
                    "texts": texts,
                    "provider": choice.get("provider", ""),
                    "model": choice.get("model", ""),
                },
            )
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
        "translation_model": choice.get("model"),
        "skipped_translation": source == target,
        "segment_count": len(job["segments"]),
    }


@app.post("/stages/synthesize")
def synthesize(request: JobRequest) -> dict:
    reused = _reuse(request.job_id, "synthesize", request.force)
    if reused:
        return reused
    with _stage(request.job_id, "synthesize") as (job, folder):
        target = job["config"]["target_language"]
        choice = job["config"].get("tts") or {}
        output_dir = folder / "tts"
        output_dir.mkdir(exist_ok=True)

        models: Dict[str, int] = {}
        for segment in job["segments"]:
            text = (segment.get("translated_text") or "").strip()
            if not text:
                continue
            data = {
                "text": text,
                "language": target,
                "speed": "1.0",
                "provider": choice.get("provider", ""),
                "model": choice.get("model", ""),
            }
            data["speaker_id"] = segment.get("speaker_id") or ""
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
            voice = response.headers.get("x-tts-voice")
            if voice:
                segment["tts_voice"] = voice
                speaker = segment.get("speaker_id")
                if speaker:
                    job.setdefault("speaker_voice_map", {})[speaker] = voice
            models[model_name] = models.get(model_name, 0) + 1
        backend = _resolve_backend()
        job["tts_provider"] = backend[0] if backend else "unknown"
        job["models_used"] = models

    generated = sum(1 for segment in job["segments"] if segment.get("tts_file"))
    return {
        "job_id": request.job_id,
        "stage": "synthesize",
        "status": "completed",
        "generated_segments": generated,
        "tts_model": (job["config"].get("tts") or {}).get("model"),
        "provider": job["tts_provider"],
        "models_used": job.get("models_used"),
        "speaker_voice_map": job.get("speaker_voice_map"),
    }


@app.post("/stages/align")
def align_speech(request: JobRequest) -> dict:
    reused = _reuse(request.job_id, "align", request.force)
    if reused:
        return reused
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
    reused = _reuse(request.job_id, "separate", request.force)
    if reused:
        return reused
    job = _load_job(request.job_id)
    if not _features(job).get("source_separation"):
        return _skip(request.job_id, "separate", "source separation is off for this job")

    with _stage(request.job_id, "separate") as (job, folder):
        original = folder / job["files"]["original_audio"]
        choice = job["config"].get("separation") or {}
        data = {
            "provider": choice.get("provider", ""),
            "model": choice.get("model", ""),
            "stem": "background",
        }
        if ASYNC_STAGES:
            data["async_mode"] = "true"
        with original.open("rb") as stream:
            response = _colab_request(
                "/separate",
                files={"audio": (original.name, stream, "audio/wav")},
                data=data,
            )
        if response.status_code == 202:
            state = _json(response)
            _await_task(state, "/separate")
            response = _colab_request(
                f"/tasks/{state['task_id']}/download", method="GET"
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
    reused = _reuse(request.job_id, "mix", request.force)
    if reused:
        return reused
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


@app.post("/stages/render")
def render(request: JobRequest) -> dict:
    reused = _reuse(request.job_id, "render", request.force)
    if reused:
        return reused
    with _stage(request.job_id, "render") as (job, folder):
        files = job["files"]
        video = folder / files["input"]
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


JOBS_PAGE = 50


def _stalled_for(job: dict, folder: Path) -> Optional[float]:
    """Return how long a running job has been stalled, if applicable."""
    if job.get("status") != "running":
        return None
    stamp = job.get("stage_started_at")
    if stamp:
        try:
            started = datetime.fromisoformat(stamp).timestamp()
        except ValueError:
            started = (folder / "job.json").stat().st_mtime
    else:
        started = (folder / "job.json").stat().st_mtime
    idle = time.time() - started
    return idle if idle >= STALLED_AFTER else None


def _summary(job: dict, folder: Optional[Path] = None) -> dict:
    idle = _stalled_for(job, folder) if folder else None
    planned = [name for name in ["upload", *STAGES] if name not in job.get("skipped_stages", [])]
    done = [name for name in job.get("completed_stages", []) if name in planned]
    config = job.get("config") or {}
    return {
        "job_id": job["job_id"],
        "status": job.get("status"),
        "current_stage": job.get("current_stage"),
        "completed_stages": job.get("completed_stages", []),
        "skipped_stages": job.get("skipped_stages", []),
        "progress": f"{len(done)}/{len(planned)}",
        "percent": round(100 * len(done) / max(1, len(planned))),
        "source_filename": job.get("source_filename"),
        "requested_source_language": config.get("source_language"),
        "source_language": job.get("source_language"),
        "target_language": job.get("target_language") or config.get("target_language"),
        "asr_model": (config.get("asr") or {}).get("model"),
        "translation_model": (config.get("translation") or {}).get("model"),
        "tts_model": (config.get("tts") or {}).get("model"),
        "features": config.get("features") or {},
        "speaker_voice_map": job.get("speaker_voice_map"),
        "segments": len(job.get("segments") or []),
        "created_at": job.get("created_at"),
        "finished_at": job.get("finished_at"),
        "stalled_for": round(idle) if idle else None,
        "error": job.get("error"),
        "download_url": f"/jobs/{job['job_id']}/download"
        if (job.get("files") or {}).get("output") else None,
        "subtitle_url": f"/jobs/{job['job_id']}/subtitle"
        if (job.get("files") or {}).get("subtitle") else None,
    }


@app.get("/jobs")
def list_jobs(limit: int = JOBS_PAGE) -> dict:
    """Return readable jobs, newest first."""
    rows = []
    for folder in ROOT.iterdir():
        if not folder.is_dir():
            continue
        manifest = folder / "job.json"
        if not manifest.exists():
            continue
        try:
            rows.append(_summary(json.loads(manifest.read_text(encoding="utf-8")), folder))
        except (ValueError, KeyError):
            continue
    rows.sort(key=lambda row: (row.get("created_at") or "", row["job_id"]), reverse=True)
    return {"jobs": rows[: max(1, min(limit, 200))], "total": len(rows)}


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = _load_job(job_id)
    idle = _stalled_for(job, _job_dir(job_id))
    job["stalled_for"] = round(idle) if idle else None
    return job


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str) -> dict:
    shutil.rmtree(_removable_job_dir(job_id), ignore_errors=True)
    return {"job_id": job_id, "deleted": True}


@app.post("/jobs/{job_id}/start", status_code=202)
def start_job(job_id: str) -> dict:
    job = _load_job(job_id)
    completed = job.get("completed_stages", [])
    stalled = _stalled_for(job, _job_dir(job_id))
    resuming = job.get("status") == "failed" or stalled is not None
    if not resuming and (job.get("status") == "completed" or len(completed) > 1):
        return {
            "job_id": job_id,
            "status": job.get("status", "running"),
            "message": "Job has already started",
        }
    failed_stage = ((job.get("error") or {}).get("stage") or job.get("current_stage")
                    if resuming else None)
    if resuming:
        if failed_stage in completed:
            completed.remove(failed_stage)
        job["status"] = "running"
        job.pop("error", None)
        _save_job(job)

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

    return {
        "job_id": job_id,
        "status": "accepted",
        "orchestrator": "n8n",
        "resumed_from": failed_stage,
        "reusing": completed if resuming else [],
        "was_stalled": stalled is not None,
    }


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
