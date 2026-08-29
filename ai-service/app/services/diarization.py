"""Stage 4: speaker diarization with pyannote.audio 4.x.

Default model is **pyannote/speaker-diarization-community-1**, which supersedes
the legacy `speaker-diarization-3.1` + `segmentation-3.0` combo:

* it is a *complete* pipeline (segmentation + embedding + PLDA + VBx clustering
  live inside the checkpoint), so only ONE gated repo has to be accepted;
* it is measurably better at speaker assignment and speaker counting
  (e.g. AMI IHM DER 17.0 vs 18.8, AliMeeting 20.3 vs 24.5 - pyannote benchmark,
  2025-09);
* it exposes **exclusive speaker diarization**, where every instant belongs to at
  most one speaker. That is exactly what this pipeline needs, because merging
  fine-grained diarization turns with Whisper's coarser segment timestamps is
  ambiguous whenever two speakers overlap.

The legacy 3.1 pipeline still works - `_turns_of()` handles both output shapes -
so switching back is a one-line change in `.env`.

Requires a Hugging Face token. When it is missing the stage degrades to a
single-speaker assumption instead of failing the whole job.
"""
from __future__ import annotations

import logging
from typing import Any

from app.core.config import settings
from app.core.device import resolve_device_for
from app.core.errors import DiarizationError, ModelLoadError
from app.services import ffmpeg
from app.services.models_registry import get_or_load
from app.services.workspace import JobWorkspace

log = logging.getLogger(__name__)

#: Turns shorter than this are clustering noise, not speech.
_MIN_TURN_SECONDS = 0.1


def _load_pipeline():  # noqa: ANN201
    def loader():  # noqa: ANN202
        try:
            import torch
            from pyannote.audio import Pipeline
        except ImportError as exc:  # pragma: no cover
            raise ModelLoadError("pyannote.audio is not installed") from exc

        try:
            # pyannote.audio 4.x renamed the argument to `token`.
            pipeline = Pipeline.from_pretrained(
                settings.diarization_model, token=settings.huggingface_token
            )
        except TypeError:  # pragma: no cover - pyannote.audio 3.x fallback
            pipeline = Pipeline.from_pretrained(
                settings.diarization_model, use_auth_token=settings.huggingface_token
            )
        except Exception as exc:
            raise ModelLoadError(
                f"Cannot load '{settings.diarization_model}': {exc}. "
                f"Accept the conditions at https://hf.co/{settings.diarization_model} "
                "and check HUGGINGFACE_TOKEN."
            ) from exc

        if pipeline is None:
            raise ModelLoadError(
                "pyannote returned no pipeline. Accept the conditions at "
                f"https://hf.co/{settings.diarization_model} and check HUGGINGFACE_TOKEN."
            )
        device = resolve_device_for("diarization")
        if device != "cpu":
            pipeline.to(torch.device(device))
        return pipeline

    return get_or_load(f"pyannote::{settings.diarization_model}", loader)


def diarize(job_id: str, audio_key: str, min_speakers: int | None = None,
            max_speakers: int | None = None, num_speakers: int | None = None) -> dict:
    ws = JobWorkspace(job_id)

    if not settings.diarization_enabled or not settings.has_hf_token:
        reason = ("diarization disabled" if not settings.diarization_enabled
                  else "HUGGINGFACE_TOKEN not configured")
        return _single_speaker_fallback(ws, audio_key, reason)

    audio = ws.pull(audio_key)

    try:
        pipeline = _load_pipeline()
    except ModelLoadError as exc:
        return _single_speaker_fallback(ws, audio_key, f"model load failed: {exc.message}")

    kwargs: dict = {}
    if num_speakers:
        kwargs["num_speakers"] = num_speakers
    else:
        if min_speakers:
            kwargs["min_speakers"] = min_speakers
        if max_speakers:
            kwargs["max_speakers"] = max_speakers

    log.info("diarizing %s with %s %s", audio.name, settings.diarization_model, kwargs or "")
    try:
        output = pipeline(str(audio), **kwargs)
    except Exception as exc:
        raise DiarizationError(f"pyannote failed: {exc}") from exc

    # community-1 returns an object with two annotations; 3.1 returns the
    # Annotation itself. Support both.
    regular = getattr(output, "speaker_diarization", output)
    exclusive_annotation = getattr(output, "exclusive_speaker_diarization", None)

    turns = _turns_of(regular)
    exclusive_turns = _turns_of(exclusive_annotation)
    has_exclusive = bool(exclusive_turns)

    if not turns and not exclusive_turns:
        return _single_speaker_fallback(ws, audio_key, "no speaker turns returned")

    speakers = sorted({t["speaker_id"] for t in (turns or exclusive_turns)})

    payload = {
        "job_id": job_id,
        "model": settings.diarization_model,
        "speakers": speakers,
        "speaker_count": len(speakers),
        "turns": turns,
        "exclusive_turns": exclusive_turns,
        "has_exclusive": has_exclusive,
        "enabled": True,
    }
    key = ws.push_json(payload, ws.layout.diarization)

    log.info("found %d speaker(s) across %d turns (exclusive: %s)",
             len(speakers), len(turns), "yes" if has_exclusive else "no")
    return {
        "diarization_key": key,
        "speaker_count": len(speakers),
        "speakers": speakers,
        "enabled": True,
        "has_exclusive": has_exclusive,
        "model": settings.diarization_model,
    }


# ---------------------------------------------------------------- internals --
def _turns_of(annotation: Any) -> list[dict]:
    """Normalise a pyannote annotation into plain sorted dicts.

    pyannote.core exposes `itertracks(yield_label=True)`; newer releases also let
    you iterate `(segment, label)` pairs directly. Both are accepted so the same
    code runs against community-1 and the legacy 3.1 pipeline.
    """
    if annotation is None:
        return []

    pairs: list[tuple[Any, Any]] = []
    itertracks = getattr(annotation, "itertracks", None)
    if callable(itertracks):
        try:
            pairs = [(segment, label) for segment, _, label in itertracks(yield_label=True)]
        except Exception:  # pragma: no cover - fall through to plain iteration
            pairs = []

    if not pairs:
        try:
            for item in annotation:
                if isinstance(item, tuple) and len(item) == 2:
                    pairs.append((item[0], item[1]))
                elif isinstance(item, tuple) and len(item) == 3:
                    pairs.append((item[0], item[2]))
        except TypeError:  # pragma: no cover - not iterable
            return []

    turns = [
        {
            "start": round(float(segment.start), 3),
            "end": round(float(segment.end), 3),
            "duration": round(float(segment.end) - float(segment.start), 3),
            "speaker_id": str(label),
        }
        for segment, label in pairs
        if float(segment.end) - float(segment.start) > _MIN_TURN_SECONDS
    ]
    turns.sort(key=lambda t: t["start"])
    return turns


def _single_speaker_fallback(ws: JobWorkspace, audio_key: str, reason: str) -> dict:
    log.warning("diarization fallback -> single speaker (%s)", reason)
    audio = ws.pull(audio_key)
    duration = ffmpeg.duration_of(audio)
    turn = {"start": 0.0, "end": round(duration, 3), "duration": round(duration, 3),
            "speaker_id": "SPEAKER_00"}
    payload = {
        "job_id": ws.job_id,
        "model": "fallback:single-speaker",
        "speakers": ["SPEAKER_00"],
        "speaker_count": 1,
        "turns": [turn],
        "exclusive_turns": [turn],
        "has_exclusive": False,
        "enabled": False,
        "reason": reason,
    }
    key = ws.push_json(payload, ws.layout.diarization)
    return {
        "diarization_key": key,
        "speaker_count": 1,
        "speakers": ["SPEAKER_00"],
        "enabled": False,
        "has_exclusive": False,
        "reason": reason,
    }
