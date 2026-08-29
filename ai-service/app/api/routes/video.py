"""Final render endpoint."""
from __future__ import annotations

from fastapi import APIRouter

from app.core.errors import InvalidInput
from app.jobs import service as jobs
from app.jobs.db import session_scope
from app.schemas.pipeline import RenderRequest, RenderResponse
from app.services import render as render_service
from app.storage.layout import JobLayout

router = APIRouter(prefix="/video", tags=["6 - mixing & render"])


@router.post("/render", response_model=RenderResponse,
             summary="Mux the final mix onto the original video")
def render(payload: RenderRequest):
    layout = JobLayout(payload.job_id)
    with session_scope() as session:
        job = jobs.get_job(session, payload.job_id)
        video_key = payload.video_key or job.video_key
        target = payload.target_language or job.target_language

    if not video_key:
        raise InvalidInput("No source video registered for this job.")
    if not target:
        raise InvalidInput("No target language known for this job.")

    subtitle_key = payload.subtitle_key or layout.subtitle(target, "srt")

    with jobs.track(payload.job_id, "render_video") as out:
        result = render_service.render(
            payload.job_id, video_key,
            payload.audio_key or layout.final_audio,
            target_language=target,
            burn_subtitles=payload.burn_subtitles,
            subtitle_key=subtitle_key,
        )
        out.update(result)

    jobs.mark_completed(payload.job_id, output_key=result["output_key"],
                        subtitle_key=subtitle_key)
    return RenderResponse(job_id=payload.job_id, stage="render_video", **result)
