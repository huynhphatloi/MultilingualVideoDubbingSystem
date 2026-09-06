"""Shared FFmpeg audio filter graphs."""
from __future__ import annotations

from typing import List, Sequence

MIX_SAMPLE_RATE = 48000
VOICEOVER_BACKGROUND_GAIN = 0.25
VOICEOVER_DUB_GAIN = 1.5
SEPARATED_BACKGROUND_GAIN = 1.0
SEPARATED_DUB_GAIN = 1.0
LOUDNESS = "loudnorm=I=-16:TP=-1.5:LRA=11"


def atempo_chain(speed: float) -> List[str]:
    """atempo accepts 0.5-2.0 per stage, so chain them for anything wider."""
    remaining = float(speed)
    stages: List[str] = []
    while remaining > 2.0:
        stages.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        stages.append("atempo=0.5")
        remaining /= 0.5
    stages.append("atempo=%.4f" % remaining)
    return stages


def atempo_filter(speed: float) -> str:
    return ",".join(atempo_chain(speed))


def dub_filtergraph(
    starts: Sequence[float],
    total_duration: float,
    rate: int = MIX_SAMPLE_RATE,
) -> str:
    if not starts:
        raise ValueError("A dub track needs at least one clip")
    filters, labels = [], []
    for index, start in enumerate(starts):
        delay = max(0, round(float(start) * 1000))
        filters.append(
            "[%d:a]aresample=%d,asetpts=PTS-STARTPTS,adelay=%d|%d[s%d]"
            % (index, rate, delay, delay, index)
        )
        labels.append("[s%d]" % index)
    filters.append(
        "%samix=inputs=%d:duration=longest:normalize=0,"
        "apad=whole_dur=%.3f,atrim=0:%.3f[dub]"
        % ("".join(labels), len(labels), total_duration, total_duration)
    )
    return ";".join(filters)


def blend_filtergraph(separated: bool, rate: int = MIX_SAMPLE_RATE) -> str:
    background = SEPARATED_BACKGROUND_GAIN if separated else VOICEOVER_BACKGROUND_GAIN
    dub = SEPARATED_DUB_GAIN if separated else VOICEOVER_DUB_GAIN
    return (
        "[0:a]aresample=%d,volume=%s[background];[1:a]aresample=%d,volume=%s[voice];"
        "[background][voice]amix=inputs=2:duration=first:normalize=0,%s[out]"
        % (rate, background, rate, dub, LOUDNESS)
    )


def mix_mode(separated: bool) -> str:
    return "separated" if separated else "voice-over"
