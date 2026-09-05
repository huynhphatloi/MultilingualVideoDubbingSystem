"""Audio filter graphs, shared by both implementations of the pipeline.

The Colab service and the local FFmpeg service each lay the generated speech
onto a track and blend it with the original. They have to agree on how, down to
the gains: two copies of these numbers is how a dub ends up sounding different
depending on which route produced it. The strings are built here and executed
by whoever needs them.
"""
from __future__ import annotations

from typing import List, Sequence

MIX_SAMPLE_RATE = 48000
#: Without source separation the original speech is still there, so it is
#: pushed well under the dub - a voice-over.
VOICEOVER_BACKGROUND_GAIN = 0.25
VOICEOVER_DUB_GAIN = 1.5
#: With the speech stem removed there is nothing to talk over, so music and
#: effects keep their level.
SEPARATED_BACKGROUND_GAIN = 1.0
SEPARATED_DUB_GAIN = 1.0
#: Broadcast-ish target, applied once at the end.
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
    """Place one generated clip per input at its own timestamp.

    Every clip is resampled first: they come from whichever engine produced
    them, at whatever rate that engine likes.
    """
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
    """Mix the background (input 0) with the dub track (input 1).

    The background is resampled explicitly: a separation stem comes back at the
    separation model's own rate, which is not the mix rate.
    """
    background = SEPARATED_BACKGROUND_GAIN if separated else VOICEOVER_BACKGROUND_GAIN
    dub = SEPARATED_DUB_GAIN if separated else VOICEOVER_DUB_GAIN
    return (
        "[0:a]aresample=%d,volume=%s[background];[1:a]aresample=%d,volume=%s[voice];"
        "[background][voice]amix=inputs=2:duration=first:normalize=0,%s[out]"
        % (rate, background, rate, dub, LOUDNESS)
    )


def mix_mode(separated: bool) -> str:
    return "separated" if separated else "voice-over"
