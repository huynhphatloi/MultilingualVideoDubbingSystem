"""Voice-activity windows for recognisers without timestamps."""
from __future__ import annotations

from pathlib import Path
from typing import List

from core.errors import MissingDependency, ProviderFailure

SAMPLE_RATE = 16000


def speech_windows(
    audio: Path,
    min_silence_ms: int = 500,
    speech_pad_ms: int = 200,
    max_speech_s: float = 20.0,
) -> List["object"]:
    from providers.asr.base import Window

    try:
        from faster_whisper.audio import decode_audio
        from faster_whisper.vad import VadOptions, get_speech_timestamps
    except ImportError as exc:  # pragma: no cover - base install always has it
        raise MissingDependency("faster-whisper", "Voice-activity windowing") from exc

    samples = decode_audio(str(audio), sampling_rate=SAMPLE_RATE)
    options = VadOptions(
        min_silence_duration_ms=min_silence_ms,
        speech_pad_ms=speech_pad_ms,
        max_speech_duration_s=max_speech_s,
    )
    chunks = get_speech_timestamps(samples, options, sampling_rate=SAMPLE_RATE)
    if not chunks:
        raise ProviderFailure("Voice-activity detection found no speech in the audio")
    return [
        Window(round(chunk["start"] / SAMPLE_RATE, 3), round(chunk["end"] / SAMPLE_RATE, 3))
        for chunk in chunks
    ]
