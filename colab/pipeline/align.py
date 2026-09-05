"""Fit each generated line into the gap it has to fill.

Optional, on by default. A line that runs long is sped up as far as the
configured limit allows; if that is not enough the overrun is recorded rather
than hidden, so a caller can see which lines will overlap.
"""
from __future__ import annotations

from pathlib import Path

from core.config import rebuild
from core.media import Scratch, duration, retime
from dubflow_core import alignment as align

NAME = "align"


def enabled(job: dict) -> bool:
    return bool(job.get("config", {}).get("features", {}).get("alignment"))


def run(job: dict, folder: Path) -> None:
    config = rebuild(job)
    segments = job["segments"]
    plans = align.plan_segments(segments, job.get("duration_seconds"), config.limits)

    for segment, plan in zip(segments, plans):
        path = segment.get("tts_file")
        if not path:
            align.record(segment, align.Plan(1.0, "unmeasured", plan.window, 0.0), None)
            continue
        clip = folder / path
        if abs(plan.speed - 1.0) <= 1e-3:
            align.record(segment, plan, segment.get("tts_duration_raw"))
            continue
        with Scratch(".wav") as scratch:
            retime(clip, scratch, plan.speed)
            scratch.replace(clip)
        align.record(segment, plan, duration(clip))

    job["alignment"] = align.summary(segments)
