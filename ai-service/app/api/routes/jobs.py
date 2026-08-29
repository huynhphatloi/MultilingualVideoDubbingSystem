"""Job lifecycle, uploads, artifact access and human review."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Query, Request, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.core import languages
from app.core.config import settings
from app.core.errors import ArtifactNotFound, InvalidInput
from app.jobs import service as jobs
from app.jobs.db import get_session, session_scope
from app.jobs.models import PIPELINE_STAGES
from app.schemas.pipeline import (
    CreateJobRequest,
    CreateJobResponse,
    RegisterVideoRequest,
    ReviewTranslationRequest,
)
from app.services import ffmpeg
from app.services.translation import service as translation_service
from app.services.workspace import JobWorkspace
from app.storage import get_storage
from app.storage.layout import JobLayout

log = logging.getLogger(__name__)
router = APIRouter(tags=["jobs"])

_ALLOWED_VIDEO = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v", ".mpg", ".mpeg"}


# ------------------------------------------------------------------ create ---
@router.post("/jobs", response_model=CreateJobResponse, summary="Create a processing job")
def create_job(payload: CreateJobRequest, session: Session = Depends(get_session)):
    target = languages.normalize(payload.target_language)
    if not target:
        raise InvalidInput(f"Unsupported target language '{payload.target_language}'",
                           details={"supported": [x["code"] for x in languages.catalog()]})
    source = languages.normalize(payload.source_language) if payload.source_language else None

    # create_job commits before returning; the "create_job" stage is closed
    # inside that same transaction.
    job = jobs.create_job(
        session,
        source_filename=payload.source_filename,
        source_language=source,
        target_language=target,
        options=payload.options,
    )
    return CreateJobResponse(
        job_id=job.id, status="created", prefix=JobLayout(job.id).prefix,
        target_language=target, source_language=source,
    )


@router.post("/jobs/{job_id}/upload", summary="Upload the source video into storage")
def upload_video(job_id: str, file: UploadFile = File(...),
                 target_language: str | None = Form(None)):
    suffix = Path(file.filename or "input.mp4").suffix.lower() or ".mp4"
    if suffix not in _ALLOWED_VIDEO:
        raise InvalidInput(f"Unsupported container '{suffix}'",
                           details={"allowed": sorted(_ALLOWED_VIDEO)})

    with jobs.track(job_id, "store_video") as out:
        ws = JobWorkspace(job_id)
        local = ws.path("source", f"input{suffix}")
        with local.open("wb") as fh:
            shutil.copyfileobj(file.file, fh, length=8 * 1024 * 1024)
        file.file.close()

        info = ffmpeg.media_info(local)
        key = ws.push(local, ws.layout.source_video(suffix.lstrip(".")))

        updates = {
            "video_key": key,
            "source_filename": file.filename,
            "duration_seconds": info["duration"],
        }
        if target_language:
            normalized = languages.normalize(target_language)
            if normalized:
                updates["target_language"] = normalized
        jobs.update_job(job_id, **updates)

        out.update({"video_key": key, "duration": info["duration"],
                    "size_bytes": info["size_bytes"], "media_info": info})

    return {"job_id": job_id, "video_key": out["video_key"], "media_info": out["media_info"]}


@router.post("/jobs/{job_id}/video", summary="Register a video already present in storage")
def register_video(job_id: str, payload: RegisterVideoRequest):
    storage = get_storage()
    if not storage.exists(payload.video_key):
        raise ArtifactNotFound(f"No such object: {payload.video_key}")
    with jobs.track(job_id, "store_video") as out:
        ws = JobWorkspace(job_id)
        local = ws.pull(payload.video_key)
        info = ffmpeg.media_info(local)
        jobs.update_job(job_id, video_key=payload.video_key,
                        duration_seconds=info["duration"])
        out.update({"video_key": payload.video_key, "duration": info["duration"]})
    return {"job_id": job_id, "video_key": payload.video_key}


# ------------------------------------------------------------------- read ----
@router.get("/jobs", summary="List jobs")
def list_jobs(limit: int = Query(50, le=200), offset: int = 0, status: str | None = None,
              session: Session = Depends(get_session)):
    items = jobs.list_jobs(session, limit=limit, offset=offset, status=status)
    return {"jobs": [j.to_dict(include_stages=True) for j in items], "count": len(items)}


@router.get("/jobs/{job_id}", summary="Job status, progress and stage history")
def get_job(job_id: str, session: Session = Depends(get_session)):
    return jobs.get_job(session, job_id).to_dict()


@router.get("/jobs/{job_id}/segments", summary="Working document (transcript + translation)")
def get_segments(job_id: str):
    ws = JobWorkspace(job_id)
    storage = get_storage()
    if not storage.exists(ws.layout.segments):
        raise ArtifactNotFound("Segments are not available yet for this job.",
                               details={"job_id": job_id})
    return storage.get_json(ws.layout.segments)


@router.get("/jobs/{job_id}/artifacts", summary="Every object produced for this job")
def list_artifacts(job_id: str):
    storage = get_storage()
    layout = JobLayout(job_id)
    keys = storage.list(layout.prefix)
    return {
        "job_id": job_id,
        "count": len(keys),
        "artifacts": [
            {"key": k, "size": _safe_size(storage, k), "url": storage.url(k)} for k in keys
        ],
    }


@router.get("/jobs/{job_id}/download/{kind}", summary="Redirect-free download of a result")
def download(job_id: str, kind: str, session: Session = Depends(get_session)):
    job = jobs.get_job(session, job_id)
    layout = JobLayout(job_id)
    lang = job.target_language or "out"
    mapping = {
        "video": job.output_key or layout.output_video(lang),
        "subtitle": job.subtitle_key or layout.subtitle(lang, "srt"),
        "subtitle-vtt": layout.subtitle(lang, "vtt"),
        "audio": layout.final_audio,
        "transcript": layout.segments,
    }
    key = mapping.get(kind)
    if not key:
        raise InvalidInput(f"Unknown download kind '{kind}'",
                           details={"available": sorted(mapping)})
    return _stream(key)


@router.get("/artifacts/{key:path}", summary="Stream any artifact by object key")
def get_artifact(key: str):
    return _stream(key)


# ----------------------------------------------------------------- review ----
@router.post("/jobs/{job_id}/review", summary="Apply human edits to the translation")
def review(job_id: str, payload: ReviewTranslationRequest):
    result = translation_service.apply_review(job_id, payload.edits)
    jobs.update_job(job_id, metrics={"human_edits": result["count"]})
    return {"job_id": job_id, **result}


@router.post("/jobs/{job_id}/complete", summary="Mark the job finished")
def complete(job_id: str, output_key: str | None = None, subtitle_key: str | None = None):
    jobs.mark_completed(job_id, output_key=output_key, subtitle_key=subtitle_key)
    return {"job_id": job_id, "status": "completed"}


@router.post("/jobs/{job_id}/fail", summary="Mark the job failed (used by n8n error branch)")
async def fail(job_id: str, request: Request):
    body = await request.json() if await request.body() else {}
    code = str(body.get("code") or "workflow_error")
    message = str(body.get("message") or "Workflow reported a failure")
    jobs.mark_failed(job_id, code, message)
    return {"job_id": job_id, "status": "failed", "code": code}


@router.delete("/jobs/{job_id}", summary="Delete a job and all its artifacts")
def delete_job(job_id: str):
    storage = get_storage()
    removed = storage.delete_prefix(JobLayout(job_id).prefix)
    with session_scope() as session:
        session.delete(jobs.get_job(session, job_id))
    shutil.rmtree(Path(settings.scratch_root) / job_id, ignore_errors=True)
    return {"job_id": job_id, "deleted_objects": removed}


@router.get("/pipeline/stages", tags=["metadata"], summary="Canonical stage order")
def pipeline_stages():
    return {"stages": list(PIPELINE_STAGES)}


# -------------------------------------------------------------- internals ----
def _stream(key: str):  # noqa: ANN202
    storage = get_storage()
    if not storage.exists(key):
        raise ArtifactNotFound(f"No such object: {key}", details={"key": key})
    filename = key.rsplit("/", 1)[-1]
    return StreamingResponse(
        storage.stream(key),
        media_type=storage.content_type_for(key),
        headers={"Content-Disposition": f'inline; filename="{filename}"',
                 "Content-Length": str(storage.size(key))},
    )


def _safe_size(storage, key: str) -> int:  # noqa: ANN001
    try:
        return storage.size(key)
    except Exception:  # noqa: BLE001
        return 0

