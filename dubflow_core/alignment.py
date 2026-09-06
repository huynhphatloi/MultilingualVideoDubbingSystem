"""Speech-duration alignment calculations."""
from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional, Sequence

DEFAULT_MIN_SPEED = 0.75
DEFAULT_MAX_SPEED = 1.35
DEFAULT_TOLERANCE = 0.05


class Limits(NamedTuple):
    min_speed: float = DEFAULT_MIN_SPEED
    max_speed: float = DEFAULT_MAX_SPEED
    tolerance: float = DEFAULT_TOLERANCE
    allow_stretch: bool = False


class Plan(NamedTuple):
    speed: float
    status: str
    window: float
    overflow: float


def window_for(
    start: float,
    end: float,
    next_start: Optional[float],
    media_duration: Optional[float],
) -> float:
    """Return the available duration before the next segment."""
    limit = next_start if next_start is not None else media_duration
    if limit is None:
        return max(0.0, float(end) - float(start))
    return max(float(end) - float(start), float(limit) - float(start), 0.0)


def plan(tts_duration: Optional[float], window: float, limits: Limits = Limits()) -> Plan:
    if tts_duration is None or tts_duration <= 0:
        return Plan(1.0, "unmeasured", round(window, 3), 0.0)
    if window <= 0:
        return Plan(1.0, "unmeasured", 0.0, round(tts_duration, 3))

    ratio = tts_duration / window
    if abs(ratio - 1.0) <= limits.tolerance:
        return Plan(1.0, "fits", round(window, 3), 0.0)

    if ratio < 1.0:
        if not limits.allow_stretch:
            return Plan(1.0, "fits", round(window, 3), 0.0)
        speed = max(limits.min_speed, ratio)
        return Plan(round(speed, 3), "stretched", round(window, 3), 0.0)

    speed = min(limits.max_speed, ratio)
    final = tts_duration / speed
    overflow = max(0.0, final - window)
    status = "clamped" if overflow > limits.tolerance else "aligned"
    return Plan(round(speed, 3), status, round(window, 3), round(overflow, 3))


def plan_segments(
    segments: Sequence[Dict],
    media_duration: Optional[float] = None,
    limits: Limits = Limits(),
) -> List[Plan]:
    ordered = sorted(range(len(segments)), key=lambda index: float(segments[index]["start"]))
    plans: List[Plan] = [Plan(1.0, "unmeasured", 0.0, 0.0)] * len(segments)
    for position, index in enumerate(ordered):
        segment = segments[index]
        following = segments[ordered[position + 1]] if position + 1 < len(ordered) else None
        next_start = float(following["start"]) if following is not None else None
        window = window_for(
            float(segment["start"]), float(segment["end"]), next_start, media_duration
        )
        plans[index] = plan(segment.get("tts_duration_raw") or segment.get("tts_duration"),
                            window, limits)
    return plans


def record(segment: Dict, chosen: Plan, final_duration: Optional[float]) -> Dict:
    source_duration = round(float(segment["end"]) - float(segment["start"]), 3)
    segment["source_duration"] = source_duration
    segment["alignment_window"] = chosen.window
    segment["alignment_speed"] = chosen.speed
    segment["alignment_status"] = chosen.status
    segment["alignment_overflow"] = chosen.overflow
    if final_duration is not None:
        segment["tts_duration_final"] = round(float(final_duration), 3)
        segment["tts_duration"] = round(float(final_duration), 3)
    return segment


def summary(segments: Sequence[Dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for segment in segments:
        status = segment.get("alignment_status")
        if status:
            counts[status] = counts.get(status, 0) + 1
    return counts
