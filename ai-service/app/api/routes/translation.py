"""Multilingual translation endpoints (n8n group 3)."""
from __future__ import annotations

from fastapi import APIRouter

from app.jobs import service as jobs
from app.jobs.db import session_scope
from app.schemas.pipeline import (
    AdaptTranslationRequest,
    AdaptTranslationResponse,
    TranslateRequest,
    TranslateResponse,
)
from app.services.translation import service as translation_service
from app.storage.layout import JobLayout

router = APIRouter(prefix="/translation", tags=["3 - translation"])


@router.post("/translate", response_model=TranslateResponse,
             summary="Duration-aware translation of every segment")
def translate(payload: TranslateRequest):
    layout = JobLayout(payload.job_id)
    target = payload.target_language or _job_target(payload.job_id)
    with jobs.track(payload.job_id, "translate") as out:
        result = translation_service.translate_job(
            payload.job_id,
            payload.segments_key or layout.segments,
            target_language=target,
            source_language=payload.source_language,
            engine=payload.engine,
            duration_aware_mode=payload.duration_aware,
        )
        out.update({k: result[k] for k in ("translation_key", "engine", "segment_count")})
    jobs.update_job(payload.job_id, target_language=result["target_language"],
                    source_language=result["source_language"])
    return TranslateResponse(job_id=payload.job_id, stage="translate", **result)


@router.post("/adapt", response_model=AdaptTranslationResponse,
             summary="Rewrite translations whose generated speech missed its slot")
def adapt(payload: AdaptTranslationRequest):
    layout = JobLayout(payload.job_id)
    with jobs.track(payload.job_id, "translate") as out:
        result = translation_service.adapt(
            payload.job_id,
            payload.segments_key or layout.segments,
            segment_ids=payload.segment_ids,
            engine=payload.engine,
        )
        out.update({"adapted": result["adapted"]})
    return AdaptTranslationResponse(job_id=payload.job_id, stage="adapt_translation", **{
        k: v for k, v in result.items() if k in AdaptTranslationResponse.model_fields
    })


def _job_target(job_id: str) -> str:
    with session_scope() as session:
        job = jobs.get_job(session, job_id)
        if not job.target_language:
            from app.core.errors import InvalidInput

            raise InvalidInput("Job has no target language; pass target_language explicitly.")
        return job.target_language
