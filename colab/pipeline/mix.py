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
from core.media import MIX_SAMPLE_RATE, ffmpeg
from dubflow_core.segments import voiced

NAME = "mix"

#: The original speech is left audible but well under the dub.
VOICEOVER_BACKGROUND_GAIN = 0.25
VOICEOVER_DUB_GAIN = 1.5
#: With the speech stem removed there is nothing to talk over, so the music and
#: effects keep their level.
SEPARATED_BACKGROUND_GAIN = 1.0
SEPARATED_DUB_GAIN = 1.0


def run(job: dict, folder: Path) -> None:
    segments = voiced(job["segments"])
    if not segments:
        raise ProviderFailure("No generated speech is available to mix")

    dubbed = folder / "dubbed.wav"
    command: List[str] = []
    for segment in segments:
        command.extend(["-i", str(folder / segment["tts_file"])])

    filters, labels = [], []
    for index, segment in enumerate(segments):
        delay = max(0, round(float(segment["start"]) * 1000))
        filters.append(
            f"[{index}:a]aresample={MIX_SAMPLE_RATE},asetpts=PTS-STARTPTS,"
            f"adelay={delay}|{delay}[s{index}]"
        )
        labels.append(f"[s{index}]")
    total = float(job["duration_seconds"])
    filters.append(
        f"{''.join(labels)}amix=inputs={len(labels)}:duration=longest:normalize=0,"
        f"apad=whole_dur={total:.3f},atrim=0:{total:.3f}[dub]"
    )
    ffmpeg([
        *command, "-filter_complex", ";".join(filters), "-map", "[dub]",
        "-ar", str(MIX_SAMPLE_RATE), "-ac", "2", "-c:a", "pcm_s16le", str(dubbed),
    ])

    background_name = job["files"].get("background_audio")
    separated = bool(background_name)
    background = folder / (background_name or job["files"]["original_audio"])
    background_gain = SEPARATED_BACKGROUND_GAIN if separated else VOICEOVER_BACKGROUND_GAIN
    dub_gain = SEPARATED_DUB_GAIN if separated else VOICEOVER_DUB_GAIN

    final_audio = folder / "final.wav"
    ffmpeg([
        "-i", str(background), "-i", str(dubbed),
        "-filter_complex",
        f"[0:a]volume={background_gain}[background];[1:a]volume={dub_gain}[voice];"
        f"[background][voice]amix=inputs=2:duration=first:normalize=0,"
        f"loudnorm=I=-16:TP=-1.5:LRA=11[out]",
        "-map", "[out]", "-ar", str(MIX_SAMPLE_RATE), "-ac", "2",
        "-c:a", "pcm_s16le", str(final_audio),
    ])
    job["files"].update({"dubbed_audio": dubbed.name, "final_audio": final_audio.name})
    job["mix_mode"] = "separated" if separated else "voice-over"
