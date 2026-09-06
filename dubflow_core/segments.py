"""Canonical segment schema and diarization helpers."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

DEFAULT_SPEAKER = "SPEAKER_00"

SEGMENT_KEYS = (
    "id",
    "speaker_id",
    "start",
    "end",
    "duration",
    "source_text",
    "translated_text",
    "tts_file",
)


def make_segment(
    index: int,
    start: float,
    end: float,
    source_text: str,
    speaker_id: str = DEFAULT_SPEAKER,
) -> Dict:
    start = round(float(start), 3)
    end = round(float(end), 3)
    return {
        "id": int(index),
        "speaker_id": speaker_id,
        "start": start,
        "end": end,
        "duration": round(max(0.0, end - start), 3),
        "source_text": source_text.strip(),
        "translated_text": None,
        "tts_file": None,
    }


def renumber(segments: Sequence[Dict]) -> List[Dict]:
    ordered = sorted(segments, key=lambda segment: (segment["start"], segment["end"]))
    for index, segment in enumerate(ordered):
        segment["id"] = index
    return list(ordered)


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _distance(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    if a_end < b_start:
        return b_start - a_end
    if b_end < a_start:
        return a_start - b_end
    return 0.0


def assign_speakers(
    segments: Sequence[Dict],
    turns: Sequence[Dict],
    max_gap: float = 2.0,
) -> List[Dict]:
    """Assign each segment to its strongest overlapping speaker turn."""
    if not turns:
        for segment in segments:
            segment.setdefault("speaker_id", DEFAULT_SPEAKER)
        return list(segments)

    for segment in segments:
        start, end = float(segment["start"]), float(segment["end"])
        totals: Dict[str, float] = {}
        for turn in turns:
            shared = _overlap(start, end, float(turn["start"]), float(turn["end"]))
            if shared > 0:
                totals[turn["speaker_id"]] = totals.get(turn["speaker_id"], 0.0) + shared

        span = max(1e-6, end - start)
        if totals:
            speaker = max(totals, key=lambda name: totals[name])
            segment["speaker_id"] = speaker
            segment["speaker_confidence"] = round(min(1.0, totals[speaker] / span), 3)
            continue

        nearest = min(
            turns,
            key=lambda turn: _distance(start, end, float(turn["start"]), float(turn["end"])),
        )
        gap = _distance(start, end, float(nearest["start"]), float(nearest["end"]))
        if gap <= max_gap:
            segment["speaker_id"] = nearest["speaker_id"]
            segment["speaker_confidence"] = 0.0
        else:
            segment["speaker_id"] = DEFAULT_SPEAKER
            segment["speaker_confidence"] = 0.0
    return list(segments)


def speakers_of(segments: Iterable[Dict]) -> List[str]:
    seen: Dict[str, None] = {}
    for segment in segments:
        seen.setdefault(segment.get("speaker_id") or DEFAULT_SPEAKER, None)
    return list(seen)


def merge_turns(turns: Sequence[Dict], gap: float = 0.25) -> List[Dict]:
    """Join nearby consecutive turns from the same speaker."""
    ordered = sorted(turns, key=lambda turn: (float(turn["start"]), float(turn["end"])))
    merged: List[Dict] = []
    for turn in ordered:
        start, end = round(float(turn["start"]), 3), round(float(turn["end"]), 3)
        speaker = turn["speaker_id"]
        if merged and merged[-1]["speaker_id"] == speaker and start - merged[-1]["end"] <= gap:
            merged[-1]["end"] = max(merged[-1]["end"], end)
            continue
        merged.append({"speaker_id": speaker, "start": start, "end": end})
    for turn in merged:
        turn["duration"] = round(turn["end"] - turn["start"], 3)
    return merged


def subtitle(segments: Sequence[Dict]) -> str:
    blocks: List[str] = []
    for index, segment in enumerate(sorted(segments, key=lambda item: item["start"]), start=1):
        blocks.extend([
            str(index),
            "%s --> %s" % (timestamp(segment["start"]), timestamp(segment["end"])),
            (segment.get("translated_text") or segment.get("source_text") or "").strip(),
            "",
        ])
    return "\n".join(blocks)


def timestamp(seconds: float) -> str:
    milliseconds = max(0, round(float(seconds) * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return "%02d:%02d:%02d,%03d" % (hours, minutes, secs, millis)


def voiced(segments: Iterable[Dict]) -> List[Dict]:
    return [segment for segment in segments if segment.get("tts_file")]


def public_speaker_summary(segments: Sequence[Dict]) -> List[Dict]:
    summary: Dict[str, Dict] = {}
    for segment in segments:
        name = segment.get("speaker_id") or DEFAULT_SPEAKER
        entry = summary.setdefault(name, {"speaker_id": name, "segments": 0, "seconds": 0.0})
        entry["segments"] += 1
        entry["seconds"] = round(entry["seconds"] + float(segment.get("duration") or 0.0), 3)
    return list(summary.values())


def optional_float(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(float(value), 3)
