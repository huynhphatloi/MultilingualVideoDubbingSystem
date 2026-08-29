"""Job registry operations. Every pipeline stage reports through here."""
from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.core.errors import JobNotFound, PipelineError
from app.core.logging import stage_context
from app.jobs.db import session_scope
from app.jobs.models import PIPELINE_STAGES, Job, JobStage, JobStatus, StageStatus

log = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------- queries ---
def get_job(session: Session, job_id: str) -> Job:
    job = session.execute(
        select(Job).options(selectinload(Job.stages)).where(Job.id == job_id)
    ).scalar_one_or_none()
    if job is None:
        raise JobNotFound(f"Job '{job_id}' does not exist.", details={"job_id": job_id})
    return job


def list_jobs(session: Session, limit: int = 50, offset: int = 0,
              status: str | None = None) -> list[Job]:
    stmt = select(Job).options(selectinload(Job.stages)).order_by(Job.created_at.desc())
    if status:
        stmt = stmt.where(Job.status == JobStatus(status))
    return list(session.execute(stmt.limit(limit).offset(offset)).scalars())


# ----------------------------------------------------------------- create ---
def create_job(session: Session, *, source_filename: str | None = None,
               source_language: str | None = None, target_language: str | None = None,
               options: dict[str, Any] | None = None) -> Job:
    job = Job(
        source_filename=source_filename,
        source_language=source_language,
        target_language=target_language,
        options=options or {},
        metrics={},
        status=JobStatus.CREATED,
    )
    session.add(job)
    session.flush()

    now = _utcnow()
    for index, name in enumerate(PIPELINE_STAGES):
        stage = JobStage(job_id=job.id, name=name, sequence=index)
        # The first stage IS this function - close it here, in the same
        # transaction. Doing it from a second session would read a `jobs` row
        # that has not been committed yet and blow up on the foreign key.
        if name == "create_job":
            stage.status = StageStatus.COMPLETED
            stage.started_at = now
            stage.finished_at = now
            stage.duration_ms = 0
            stage.output = {"target_language": target_language,
                            "source_language": source_language}
        session.add(stage)

    session.flush()
    # Make the row visible to the independent sessions that every later stage
    # opens (jobs.track, update_job, ...). Without this the client can POST
    # /jobs/{id}/upload before this transaction lands.
    session.commit()
    session.refresh(job)
    log.info("job created", extra={"job_id": job.id, "target_language": target_language})
    return job


# ----------------------------------------------------------------- update ---
def update_job(job_id: str, **fields: Any) -> dict:
    """Patch top-level job fields (merges dicts for options/metrics)."""
    with session_scope() as session:
        job = get_job(session, job_id)
        for key, value in fields.items():
            if value is None:
                continue
            if key in ("options", "metrics") and isinstance(value, dict):
                merged = dict(getattr(job, key) or {})
                merged.update(value)
                setattr(job, key, merged)
            elif hasattr(job, key):
                setattr(job, key, value)
        job.updated_at = _utcnow()
        session.flush()
        return job.to_dict()


def job_option(job_id: str, key: str, default: Any = None) -> Any:
    """Read one entry from a job's stored options.

    Choices the operator made when creating the job (which TTS model, whether
    to burn subtitles) live on the job, not in the per-stage request. That way
    they survive the whole run without n8n having to carry them through
    thirteen HTTP calls, and a job re-run by hand picks up the same settings.
    Missing job or missing key both return `default` - a stage must not fail
    because an option was never set.
    """
    try:
        with session_scope() as session:
            return (get_job(session, job_id).options or {}).get(key, default)
    except Exception:  # noqa: BLE001 - an option is never worth failing a stage
        return default


def mark_failed(job_id: str, code: str, message: str) -> None:
    with session_scope() as session:
        job = get_job(session, job_id)
        job.status = JobStatus.FAILED
        job.error_code = code
        job.error_message = message[:4000]
        job.updated_at = _utcnow()


def mark_completed(job_id: str, output_key: str | None = None,
                   subtitle_key: str | None = None) -> None:
    with session_scope() as session:
        job = get_job(session, job_id)
        job.status = JobStatus.COMPLETED
        job.error_code = None
        job.error_message = None
        if output_key:
            job.output_key = output_key
        if subtitle_key:
            job.subtitle_key = subtitle_key
        job.updated_at = _utcnow()


# ----------------------------------------------------------------- stages ---
def _truncate(details: dict | None, limit: int = 4000) -> dict:
    """Keep stored diagnostics bounded - stderr can be enormous."""
    if not details:
        return {}
    out: dict[str, Any] = {}
    for key, value in details.items():
        text = value if isinstance(value, str) else repr(value)
        out[key] = text[-limit:] if len(text) > limit else value
    return out


def _get_stage(session: Session, job_id: str, name: str) -> JobStage:
    stage = session.execute(
        select(JobStage).where(JobStage.job_id == job_id, JobStage.name == name)
    ).scalar_one_or_none()
    if stage is None:
        stage = JobStage(
            job_id=job_id, name=name,
            sequence=PIPELINE_STAGES.index(name) if name in PIPELINE_STAGES else 99,
        )
        session.add(stage)
        session.flush()
    return stage


def start_stage(job_id: str, name: str) -> None:
    with session_scope() as session:
        job = get_job(session, job_id)
        stage = _get_stage(session, job_id, name)
        if stage.status in (StageStatus.COMPLETED, StageStatus.FAILED):
            stage.attempt += 1
        stage.status = StageStatus.RUNNING
        stage.started_at = _utcnow()
        stage.finished_at = None
        stage.error_code = None
        stage.message = None
        if job.status in (JobStatus.CREATED, JobStatus.AWAITING_REVIEW, JobStatus.FAILED):
            job.status = JobStatus.RUNNING
            job.error_code = None
            job.error_message = None
        job.updated_at = _utcnow()


def complete_stage(job_id: str, name: str, output: dict | None = None,
                   message: str | None = None, duration_ms: int | None = None) -> None:
    with session_scope() as session:
        stage = _get_stage(session, job_id, name)
        stage.status = StageStatus.COMPLETED
        stage.finished_at = _utcnow()
        if stage.started_at and duration_ms is None:
            duration_ms = int((stage.finished_at - stage.started_at).total_seconds() * 1000)
        stage.duration_ms = duration_ms
        stage.output = output or {}
        stage.message = message
        session.get(Job, job_id).updated_at = _utcnow()



def fail_stage(job_id: str, name: str, code: str, message: str,
               details: dict | None = None) -> None:
    with session_scope() as session:
        job = get_job(session, job_id)
        stage = _get_stage(session, job_id, name)
        stage.status = StageStatus.FAILED
        stage.finished_at = _utcnow()
        stage.error_code = code
        stage.message = message[:4000]
        # Keep whatever the failing service captured (ffmpeg/demucs stderr,
        # the command line, ...). Losing it turns every failure into a guess.
        if details:
            stage.output = {"details": _truncate(details)}
        job.status = JobStatus.FAILED
        job.error_code = code
        job.error_message = message[:4000]
        job.updated_at = _utcnow()


@contextmanager
def track(job_id: str, stage: str) -> Iterator[dict]:
    """Wrap a pipeline step: timing, status transitions and error capture.

    Usage::

        with track(job_id, "extract_audio") as out:
            ...
            out["audio_key"] = key
    """
    started = time.perf_counter()
    output: dict[str, Any] = {}
    start_stage(job_id, stage)
    with stage_context(job_id=job_id, stage=stage):
        log.info("stage started")
        try:
            yield output
        except PipelineError as exc:
            fail_stage(job_id, stage, exc.code, exc.message, details=exc.details)
            log.error("stage failed: %s", exc.message,
                      extra={"error_code": exc.code, "details": _truncate(exc.details)})
            raise
        except Exception as exc:
            fail_stage(job_id, stage, "internal_error", str(exc))
            log.exception("stage crashed")
            raise
        elapsed = int((time.perf_counter() - started) * 1000)
        complete_stage(job_id, stage, output=output, duration_ms=elapsed)
        log.info("stage completed in %d ms", elapsed, extra={"duration_ms": elapsed})
