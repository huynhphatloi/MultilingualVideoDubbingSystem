"""Speech understanding + speech generation endpoints (n8n groups 2 and 4)."""
from __future__ import annotations

from fastapi import APIRouter

from app.core.config import settings
from app.jobs import service as jobs
from app.schemas.pipeline import (
    DiarizeRequest,
    DiarizeResponse,
    MergeRequest,
    MergeResponse,
    SynthesizeRequest,
    SynthesizeResponse,
    TranscribeRequest,
    TranscribeResponse,
)
from app.services import alignment, asr, diarization
from app.services.tts import router as tts_router
from app.services.tts import service as tts_service
from app.storage.layout import JobLayout

router = APIRouter(prefix="/speech", tags=["2 - speech"])


@router.get("/tts-models", tags=["metadata"],
            summary="Selectable TTS engines for a language, with why each is or is not usable")
def tts_models(language: str = "vi", has_voice_reference: bool = True) -> dict:
    """Feeds the model picker in the web UI.

    Engines that cannot serve this language come back too, each with a reason,
    so the picker can grey them out and say why instead of pretending they do
    not exist - "XTTS-v2: does not speak 'vi'" is the single most useful thing
    this API tells a new operator.
    """
    return {
        "language": language,
        "default": "auto",
        "models": tts_router.catalog(language, has_voice_reference=has_voice_reference),
    }


@router.post("/transcribe", response_model=TranscribeResponse,
             summary="Speech-to-text with faster-whisper (keeps timestamps)")
def transcribe(payload: TranscribeRequest):
    layout = JobLayout(payload.job_id)
    # ASR reads the ORIGINAL mix by default - see Settings.asr_audio_source.
    default_key = (layout.original_audio if settings.asr_audio_source == "original"
                   else layout.speech_audio)
    audio_key = payload.audio_key or default_key
    with jobs.track(payload.job_id, "transcribe") as out:
        result = asr.transcribe(
            payload.job_id, audio_key, language=payload.language,
            model=payload.model, word_timestamps=payload.word_timestamps,
        )
        out.update({k: result[k] for k in
                    ("transcript_key", "language", "language_confidence", "segment_count")})
    jobs.update_job(
        payload.job_id,
        detected_language=result["language"],
        language_confidence=result["language_confidence"],
        source_language=result["language"],
        segment_count=result["segment_count"],
    )
    return TranscribeResponse(job_id=payload.job_id, stage="transcribe", **result)


@router.post("/diarize", response_model=DiarizeResponse,
             summary="Speaker diarization with pyannote.audio")
def diarize(payload: DiarizeRequest):
    layout = JobLayout(payload.job_id)
    audio_key = payload.audio_key or layout.speech_audio
    with jobs.track(payload.job_id, "diarize") as out:
        result = diarization.diarize(
            payload.job_id, audio_key,
            min_speakers=payload.min_speakers, max_speakers=payload.max_speakers,
            num_speakers=payload.num_speakers,
        )
        out.update(result)
    jobs.update_job(payload.job_id, speaker_count=result["speaker_count"])
    return DiarizeResponse(job_id=payload.job_id, stage="diarize", **{
        k: v for k, v in result.items() if k in DiarizeResponse.model_fields
    })


@router.post("/merge", response_model=MergeResponse,
             summary="Merge transcript + speaker + timing, extract voice references")
def merge(payload: MergeRequest):
    layout = JobLayout(payload.job_id)
    with jobs.track(payload.job_id, "merge_segments") as out:
        result = alignment.merge(
            payload.job_id,
            payload.transcript_key or layout.transcript,
            payload.diarization_key or layout.diarization,
            extract_voice_references=payload.extract_voice_references,
        )
        out.update({k: result[k] for k in ("segments_key", "segment_count", "speaker_count")})
    jobs.update_job(payload.job_id, segment_count=result["segment_count"],
                    speaker_count=result["speaker_count"])
    return MergeResponse(job_id=payload.job_id, stage="merge_segments", **result)


@router.post("/synthesize", response_model=SynthesizeResponse, tags=["4 - synthesis"],
             summary="Generate target speech per segment and measure its duration")
def synthesize(payload: SynthesizeRequest):
    layout = JobLayout(payload.job_id)
    # An explicit force_model wins; otherwise fall back to what the operator
    # picked when the job was created. Putting the choice on the JOB is what
    # lets the web UI select an engine at all: n8n drives synthesis through a
    # generic call and would otherwise have to thread the parameter through
    # thirteen stages. "auto" is the picker's word for "no forcing".
    forced = payload.force_model or jobs.job_option(payload.job_id, "tts_model")
    if forced == "auto":
        forced = None
    with jobs.track(payload.job_id, "synthesize") as out:
        result = tts_service.synthesize_job(
            payload.job_id,
            payload.segments_key or layout.segments,
            target_language=payload.target_language,
            segment_ids=payload.segment_ids,
            prefer_voice_clone=payload.prefer_voice_clone,
            force_model=forced,
        )
        out.update({k: result[k] for k in
                    ("synthesized", "failed", "models_used", "ratio_stats")})
    jobs.update_job(payload.job_id, metrics={"ratio_stats": result["ratio_stats"],
                                             "tts_models": result["models_used"]})
    return SynthesizeResponse(job_id=payload.job_id, stage="synthesize", **result)
