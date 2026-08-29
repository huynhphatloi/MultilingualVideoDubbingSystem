"""Media preprocessing endpoints (called by n8n group 1)."""
from __future__ import annotations

from fastapi import APIRouter

from app.jobs import service as jobs
from app.jobs.db import session_scope
from app.schemas.pipeline import (
    ExtractAudioRequest,
    ExtractAudioResponse,
    SeparateRequest,
    SeparateResponse,
)
from app.services import media, separation
from app.storage.layout import JobLayout

router = APIRouter(prefix="/media", tags=["1 - media"])


@router.post("/extract-audio", response_model=ExtractAudioResponse,
             summary="Extract a mono 16 kHz WAV from the source video")
def extract_audio(payload: ExtractAudioRequest):
    video_key = payload.video_key or _job_video_key(payload.job_id)
    with jobs.track(payload.job_id, "extract_audio") as out:
        result = media.extract_audio(
            payload.job_id, video_key,
            sample_rate=payload.sample_rate, channels=payload.channels,
        )
        out.update({k: result[k] for k in ("audio_key", "duration_seconds", "peak_dbfs")})
    jobs.update_job(payload.job_id, duration_seconds=result["duration_seconds"])
    return ExtractAudioResponse(job_id=payload.job_id, stage="extract_audio", **{
        k: v for k, v in result.items() if k in ExtractAudioResponse.model_fields
    })


@router.post("/separate", response_model=SeparateResponse,
             summary="Split speech from music/ambience with Demucs")
def separate(payload: SeparateRequest):
    audio_key = payload.audio_key or JobLayout(payload.job_id).original_audio
    with jobs.track(payload.job_id, "separate_sources") as out:
        result = separation.separate(payload.job_id, audio_key, model=payload.model)
        out.update(result)
    return SeparateResponse(job_id=payload.job_id, stage="separate_sources", **{
        k: v for k, v in result.items() if k in SeparateResponse.model_fields
    })


def _job_video_key(job_id: str) -> str:
    with session_scope() as session:
        job = jobs.get_job(session, job_id)
        return job.video_key or JobLayout(job_id).source_video()
