"""Stage 3: speech-to-text.

Timestamps are the whole point here - they are the contract the rest of the
pipeline synchronises against.

Whisper runs either locally (faster-whisper/CTranslate2) or on the Colab GPU,
decided by REMOTE_ASR_ENABLED. Only the DECODING moves: both paths return the
same raw decoder windows, and `build_utterances` turns them into utterances
here, locally, in both cases. That split is deliberate - the segmentation
logic is pure data with its own tests, and duplicating it on the notebook side
would give two implementations to keep in agreement.

Moving it saves 1.4 GB of disk (whisper-medium) and a great deal of time: this
Mac has no CUDA, so faster-whisper falls back to CPU int8 and a 37-second clip
takes minutes. The cost is that the SPEECH TRACK is uploaded - compressed to
~24 kbps Opus, about 1.7 MB for a ten-minute film, but uploaded. See
app/services/remote_gpu.py; this is the one stage where more than text leaves
the machine.
"""
from __future__ import annotations

import logging

from app.core import languages
from app.core.config import settings
from app.core.device import resolve_compute_type, resolve_device_for
from app.core.errors import LanguageDetectionFailed, ModelLoadError, NoSpeechDetected
from app.services import ffmpeg, remote_gpu
from app.services.models_registry import get_or_load
from app.services.segmentation import build_utterances
from app.services.workspace import JobWorkspace

log = logging.getLogger(__name__)


def _load(model_name: str):  # noqa: ANN201
    def loader():  # noqa: ANN202
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover
            raise ModelLoadError("faster-whisper is not installed") from exc
        return WhisperModel(
            model_name,
            device=resolve_device_for("asr"),
            compute_type=resolve_compute_type(),
            download_root=None,
        )

    return get_or_load(f'whisper::{model_name}::' + resolve_device_for('asr'), loader)


def transcribe(job_id: str, audio_key: str, language: str | None = None,
               model: str | None = None, word_timestamps: bool = True) -> dict:
    ws = JobWorkspace(job_id)
    audio = ws.pull(audio_key)

    model_name = model or settings.whisper_model
    lang = languages.normalize(language) if language and language != "auto" else None

    if _use_remote():
        windows, detected, confidence, media_duration = _decode_remote(
            ws, audio, lang, model_name, word_timestamps)
    else:
        windows, detected, confidence, media_duration = _decode_local(
            audio, lang, model_name, word_timestamps)

    if lang is None and confidence < 0.35:
        raise LanguageDetectionFailed(
            f"Could not confidently detect the source language (best guess "
            f"'{detected}' at {confidence:.0%}). Set source_language explicitly.",
            details={"detected": detected, "confidence": confidence},
        )

    # Whisper hands back decoder windows, not utterances. Dubbing needs
    # utterances: see app/services/segmentation.py for why a raw window is
    # unusable here. This runs locally whichever backend decoded - it is pure
    # data, and one implementation is easier to trust than two.
    segments = build_utterances(
        windows,
        max_seconds=settings.asr_max_segment_seconds,
        split_gap=settings.asr_split_gap_seconds,
        min_seconds=settings.asr_min_segment_seconds,
        no_speech_threshold=settings.asr_no_speech_threshold,
        total_duration=media_duration,
    )

    if not segments:
        raise NoSpeechDetected(
            "No speech was detected in this video - there is nothing to dub.",
            details={"audio_key": audio_key, "language": detected},
        )

    log.info("whisper returned %d window(s) -> %d utterance(s) "
             "(longest %.1fs, median %.1fs)",
             len(windows), len(segments),
             max(s["duration"] for s in segments),
             sorted(s["duration"] for s in segments)[len(segments) // 2])

    payload = {
        "job_id": job_id,
        "language": detected,
        "language_confidence": round(confidence, 4),
        "model": model_name,
        "duration": round(media_duration or 0.0, 3),
        "segments": segments,
    }
    key = ws.push_json(payload, ws.layout.transcript)

    total_speech = sum(s["duration"] for s in segments)
    return {
        "transcript_key": key,
        "language": detected,
        "language_confidence": round(confidence, 4),
        "segment_count": len(segments),
        "total_speech_seconds": round(total_speech, 2),
        "preview": [
            {"start": s["start"], "end": s["end"], "text": s["source_text"]}
            for s in segments[:5]
        ],
    }


# ------------------------------------------------------------- backends ----
def _use_remote() -> bool:
    return settings.remote_asr_enabled and remote_gpu.configured()


def _decode_local(audio, lang: str | None, model_name: str,  # noqa: ANN001
                  word_timestamps: bool) -> tuple[list[dict], str, float, float | None]:
    """faster-whisper on this machine. Downloads the checkpoint on first use."""
    whisper = _load(model_name)
    log.info("transcribing locally with whisper=%s language=%s", model_name, lang or "auto")
    segments_iter, info = whisper.transcribe(
        str(audio),
        language=lang,
        beam_size=settings.whisper_beam_size,
        vad_filter=settings.whisper_vad_filter,
        vad_parameters={"min_silence_duration_ms": 500} if settings.whisper_vad_filter else None,
        word_timestamps=word_timestamps,
        condition_on_previous_text=settings.whisper_condition_on_previous_text,
    )

    windows: list[dict] = []
    for seg in segments_iter:
        text = (seg.text or "").strip()
        if not text:
            continue
        item = {
            "start": float(seg.start),
            "end": float(seg.end),
            "source_text": text,
            "avg_logprob": getattr(seg, "avg_logprob", None),
            "no_speech_prob": getattr(seg, "no_speech_prob", None),
        }
        if word_timestamps and getattr(seg, "words", None):
            item["words"] = [
                {"word": w.word, "start": round(float(w.start), 3),
                 "end": round(float(w.end), 3), "probability": getattr(w, "probability", None)}
                for w in seg.words if w.start is not None and w.end is not None
            ]
        windows.append(item)

    detected = languages.normalize(info.language) or info.language
    confidence = float(getattr(info, "language_probability", 0.0) or 0.0)
    duration = float(getattr(info, "duration", 0.0) or 0.0) or None
    return windows, detected, confidence, duration


def _decode_remote(ws, audio, lang: str | None, model_name: str,  # noqa: ANN001
                   word_timestamps: bool) -> tuple[list[dict], str, float, float | None]:
    """Whisper on the Colab GPU. Nothing is cached on this disk.

    The audio is transcoded to Opus first. That is not an optimisation to skip:
    a ten-minute film is 19 MB of wav and 1.7 MB of 24 kbps Opus, and over a
    home upstream the difference is the stage taking seconds or minutes.
    """
    compressed = ws.path("scratch", "asr_upload.opus")
    try:
        ffmpeg.to_opus(audio, compressed, kbps=settings.remote_asr_opus_kbps)
        upload = compressed
    except Exception as exc:  # noqa: BLE001 - a bigger upload beats no upload
        log.warning("could not compress the audio for upload (%s); sending wav", exc)
        upload = audio

    size_mb = upload.stat().st_size / 1_048_576
    log.info("transcribing remotely with whisper=%s language=%s (%.1f MB upload)",
             model_name, lang or "auto", size_mb)

    payload = remote_gpu.post(
        "/transcribe",
        data={
            "language": lang or "",
            "model": model_name,
            "word_timestamps": str(bool(word_timestamps)).lower(),
            "beam_size": str(settings.whisper_beam_size),
            "vad_filter": str(bool(settings.whisper_vad_filter)).lower(),
            "condition_on_previous_text":
                str(bool(settings.whisper_condition_on_previous_text)).lower(),
        },
        files={"audio": (upload.name, upload, "audio/ogg")},
        stage="remote ASR",
    )

    windows = payload.get("windows") or []
    detected = languages.normalize(payload.get("language") or "") or payload.get("language") or ""
    confidence = float(payload.get("language_confidence") or 0.0)
    duration = float(payload.get("duration") or 0.0) or None
    log.info("remote ASR returned %d window(s) in %s", len(windows), detected)
    return windows, detected, confidence, duration
