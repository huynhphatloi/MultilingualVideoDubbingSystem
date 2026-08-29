"""Subtitle generation endpoint."""
from __future__ import annotations

from fastapi import APIRouter

from app.jobs import service as jobs
from app.schemas.pipeline import SubtitleRequest, SubtitleResponse
from app.services import subtitles
from app.storage.layout import JobLayout

router = APIRouter(prefix="/subtitle", tags=["6 - mixing & render"])


@router.post("/generate", response_model=SubtitleResponse,
             summary="Produce .srt / .vtt from the translated segments")
def generate(payload: SubtitleRequest):
    layout = JobLayout(payload.job_id)
    with jobs.track(payload.job_id, "generate_subtitles") as out:
        result = subtitles.generate(
            payload.job_id, payload.segments_key or layout.segments,
            formats=list(payload.formats), include_source=payload.include_source,
            max_chars_per_line=payload.max_chars_per_line,
        )
        out.update(result)
    jobs.update_job(payload.job_id, subtitle_key=result["subtitle_keys"].get("srt"))
    return SubtitleResponse(job_id=payload.job_id, stage="generate_subtitles", **result)
