"""Pull two audio tracks out of the video.

`original.wav` is the full-quality track the mix and the voice references use;
`asr.wav` is the 16 kHz mono track every recogniser and the diarizer want.
"""
from __future__ import annotations

from pathlib import Path

from core.media import ASR_SAMPLE_RATE, MIX_SAMPLE_RATE, duration, ffmpeg

NAME = "extract"


def run(job: dict, folder: Path) -> None:
    video = folder / job["files"]["input"]
    original = folder / "original.wav"
    asr_audio = folder / "asr.wav"
    ffmpeg([
        "-i", str(video), "-vn", "-ar", str(MIX_SAMPLE_RATE), "-ac", "2",
        "-c:a", "pcm_s16le", str(original),
    ])
    ffmpeg([
        "-i", str(video), "-vn", "-ar", str(ASR_SAMPLE_RATE), "-ac", "1",
        "-c:a", "pcm_s16le", str(asr_audio),
    ])
    job["duration_seconds"] = round(duration(video), 3)
    job["files"].update({"original_audio": original.name, "asr_audio": asr_audio.name})
