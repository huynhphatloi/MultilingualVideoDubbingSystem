"""Job + stage tracking tables. This is what powers the progress UI."""
from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.jobs.db import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime) -> datetime:
    """Postgres hands back aware datetimes, SQLite naive ones. Normalise."""
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


class JobStatus(str, enum.Enum):  # noqa: UP042
    CREATED = "created"
    RUNNING = "running"
    AWAITING_REVIEW = "awaiting_review"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StageStatus(str, enum.Enum):  # noqa: UP042
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


#: Canonical pipeline stages, in execution order. The frontend renders this
#: list as the progress tracker and n8n uses the same names when reporting.
PIPELINE_STAGES: tuple[str, ...] = (
    "create_job",
    "store_video",
    "extract_audio",
    "separate_sources",
    "transcribe",
    "diarize",
    "merge_segments",
    "translate",
    "synthesize",
    "synchronize",
    "mix_audio",
    "generate_subtitles",
    "render_video",
)


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus, native_enum=False, length=32), default=JobStatus.CREATED, index=True
    )

    source_filename: Mapped[str | None] = mapped_column(String(512))
    video_key: Mapped[str | None] = mapped_column(String(512))
    output_key: Mapped[str | None] = mapped_column(String(512))
    subtitle_key: Mapped[str | None] = mapped_column(String(512))

    source_language: Mapped[str | None] = mapped_column(String(16))
    target_language: Mapped[str | None] = mapped_column(String(16))
    detected_language: Mapped[str | None] = mapped_column(String(16))
    language_confidence: Mapped[float | None] = mapped_column(Float)

    duration_seconds: Mapped[float | None] = mapped_column(Float)
    speaker_count: Mapped[int | None] = mapped_column(Integer)
    segment_count: Mapped[int | None] = mapped_column(Integer)

    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)

    options: Mapped[dict] = mapped_column(JSON, default=dict)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    stages: Mapped[list[JobStage]] = relationship(
        back_populates="job", cascade="all, delete-orphan", order_by="JobStage.sequence"
    )

    def to_dict(self, include_stages: bool = True) -> dict:
        data = {
            "job_id": self.id,
            "status": self.status.value if isinstance(self.status, JobStatus) else self.status,
            "source_filename": self.source_filename,
            "video_key": self.video_key,
            "output_key": self.output_key,
            "subtitle_key": self.subtitle_key,
            "source_language": self.source_language,
            "detected_language": self.detected_language,
            "language_confidence": self.language_confidence,
            "target_language": self.target_language,
            "duration_seconds": self.duration_seconds,
            "speaker_count": self.speaker_count,
            "segment_count": self.segment_count,
            "error": (
                {"code": self.error_code, "message": self.error_message}
                if self.error_code
                else None
            ),
            "options": self.options or {},
            "metrics": self.metrics or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
        if include_stages:
            data["stages"] = [s.to_dict() for s in self.stages]
            data["progress"] = self.progress()
        return data

    def progress(self) -> dict:
        """Progress + liveness.

        "RUNNING at 2/13" is ambiguous: it can mean a model is grinding away, or
        that nothing at all is happening because the orchestrator never fired
        the next call. The UI cannot tell those apart from a percentage, so the
        distinction is computed here instead.
        """
        done = sum(
            1 for s in self.stages
            if s.status in (StageStatus.COMPLETED, StageStatus.SKIPPED)
        )
        total = len(PIPELINE_STAGES)
        running = next((s for s in self.stages if s.status == StageStatus.RUNNING), None)

        now = _utcnow()
        idle_seconds = None
        if self.updated_at:
            idle_seconds = round((now - _as_utc(self.updated_at)).total_seconds(), 1)

        running_seconds = None
        if running and running.started_at:
            running_seconds = round((now - _as_utc(running.started_at)).total_seconds(), 1)

        terminal = self.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)

        return {
            "completed": done,
            "total": total,
            "percent": round(100 * done / total, 1) if total else 0.0,
            "current_stage": running.name if running else None,
            "running_seconds": running_seconds,
            "idle_seconds": idle_seconds,
            # No stage is executing and the job is not finished -> nobody is
            # driving it. Almost always a trigger that never reached the API.
            "waiting_for_trigger": bool(running is None and not terminal),
        }


class JobStage(Base):
    __tablename__ = "job_stages"
    __table_args__ = (Index("ix_job_stages_job_name", "job_id", "name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("jobs.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(64))
    sequence: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[StageStatus] = mapped_column(
        Enum(StageStatus, native_enum=False, length=32), default=StageStatus.PENDING
    )
    attempt: Mapped[int] = mapped_column(Integer, default=1)

    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_ms: Mapped[int | None] = mapped_column(Integer)

    message: Mapped[str | None] = mapped_column(Text)
    error_code: Mapped[str | None] = mapped_column(String(64))
    output: Mapped[dict] = mapped_column(JSON, default=dict)

    job: Mapped[Job] = relationship(back_populates="stages")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "sequence": self.sequence,
            "status": self.status.value if isinstance(self.status, StageStatus) else self.status,
            "attempt": self.attempt,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "duration_ms": self.duration_ms,
            "message": self.message,
            "error_code": self.error_code,
            "output": self.output or {},
        }
