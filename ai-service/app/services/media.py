"""Stage 1: media preprocessing (audio extraction, probing)."""
from __future__ import annotations

import logging

from app.core.errors import InvalidInput
from app.services import ffmpeg
from app.services.workspace import JobWorkspace

log = logging.getLogger(__name__)

#: Below this peak level the audio is almost certainly unusable for ASR.
QUIET_THRESHOLD_DBFS = -45.0


def extract_audio(job_id: str, video_key: str, sample_rate: int = 16000,
                  channels: int = 1) -> dict:
    ws = JobWorkspace(job_id)
    video = ws.pull(video_key)

    info = ffmpeg.media_info(video)
    if not info["has_audio"]:
        raise InvalidInput(
            "The uploaded video has no audio stream - nothing to dub.",
            details={"video_key": video_key, "format": info["format_name"]},
        )
    if info["duration"] <= 0.05:
        raise InvalidInput("Video duration is zero or unreadable.",
                           details={"video_key": video_key})

    out = ws.path("audio", "original.wav")
    ffmpeg.extract_audio(video, out, sample_rate=sample_rate, channels=channels)

    peak = ffmpeg.peak_dbfs(out)
    if peak is not None and peak < QUIET_THRESHOLD_DBFS:
        log.warning("audio is very quiet (%.1f dBFS) - ASR quality will suffer", peak)

    key = ws.push(out, ws.layout.original_audio)
    return {
        "audio_key": key,
        "duration_seconds": round(info["duration"], 3),
        "sample_rate": sample_rate,
        "channels": channels,
        "has_audio_stream": True,
        "peak_dbfs": peak,
        "video_info": info,
        "quiet_warning": bool(peak is not None and peak < QUIET_THRESHOLD_DBFS),
    }

