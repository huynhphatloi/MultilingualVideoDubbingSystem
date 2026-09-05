"""Lay the generated lines onto a single track and mix it with the original.

Two modes. Without source separation the original stays quiet underneath, which
is a voice-over - what this project has always produced. With separation the
speech stem is dropped entirely and the dub sits on the real background, which
is what a studio dub sounds like.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from core.errors import ProviderFailure
from core.media import ffmpeg
from dubflow_core import mixing
from dubflow_core.segments import voiced

NAME = "mix"


def run(job: dict, folder: Path) -> None:
    segments = voiced(job["segments"])
    if not segments:
        raise ProviderFailure("No generated speech is available to mix")

    dubbed = folder / "dubbed.wav"
    inputs: List[str] = []
    for segment in segments:
        inputs.extend(["-i", str(folder / segment["tts_file"])])
    total = float(job["duration_seconds"])
    ffmpeg([
        *inputs,
        "-filter_complex",
        mixing.dub_filtergraph([segment["start"] for segment in segments], total),
        "-map", "[dub]", "-ar", str(mixing.MIX_SAMPLE_RATE), "-ac", "2",
        "-c:a", "pcm_s16le", str(dubbed),
    ])

    background_name = job["files"].get("background_audio")
    separated = bool(background_name)
    background = folder / (background_name or job["files"]["original_audio"])

    final_audio = folder / "final.wav"
    ffmpeg([
        "-i", str(background), "-i", str(dubbed),
        "-filter_complex", mixing.blend_filtergraph(separated),
        "-map", "[out]", "-ar", str(mixing.MIX_SAMPLE_RATE), "-ac", "2",
        "-c:a", "pcm_s16le", str(final_audio),
    ])
    job["files"].update({"dubbed_audio": dubbed.name, "final_audio": final_audio.name})
    job["mix_mode"] = mixing.mix_mode(separated)
