"""Temporal synchronisation and mixing endpoints (n8n groups 5 and 6)."""
from __future__ import annotations

from fastapi import APIRouter

from app.jobs import service as jobs
from app.schemas.pipeline import (
    MixRequest,
    MixResponse,
    SynchronizeRequest,
    SynchronizeResponse,
)
from app.services import mixing, sync
from app.storage.layout import JobLayout

router = APIRouter(prefix="/audio", tags=["5 - synchronization"])


@router.post("/synchronize", response_model=SynchronizeResponse,
             summary="Place each generated take at its original timestamp")
def synchronize(payload: SynchronizeRequest):
    layout = JobLayout(payload.job_id)
    with jobs.track(payload.job_id, "synchronize") as out:
        result = sync.synchronize(
            payload.job_id, payload.segments_key or layout.segments,
            total_duration=payload.total_duration,
        )
        out.update(result)
    return SynchronizeResponse(job_id=payload.job_id, stage="synchronize", **{
        k: v for k, v in result.items() if k in SynchronizeResponse.model_fields
    })


@router.post("/mix", response_model=MixResponse, tags=["6 - mixing & render"],
             summary="Mix the dubbed voice with the preserved background stem")
def mix(payload: MixRequest):
    layout = JobLayout(payload.job_id)
    with jobs.track(payload.job_id, "mix_audio") as out:
        result = mixing.mix(
            payload.job_id,
            payload.dubbed_track_key or layout.dubbed_track,
            background_key=payload.background_key,
            background_gain_db=payload.background_gain_db,
            speech_gain_db=payload.speech_gain_db,
            loudnorm=payload.loudnorm,
            duck=payload.duck,
        )
        out.update(result)
    return MixResponse(job_id=payload.job_id, stage="mix_audio", **{
        k: v for k, v in result.items() if k in MixResponse.model_fields
    })
