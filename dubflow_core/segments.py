"""The canonical segment schema and the diarization/ASR merge.

Every stage reads and writes the same dictionary. A segment is created by the
ASR stage, gains a speaker in the merge stage, a translation, a voice file, and
finally alignment metadata - but the keys never change shape, so a stage can be
skipped without the next one having to guess.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

#: Used when diarization is off, or when a segment matches no speaker turn.
DEFAULT_SPEAKER = "SPEAKER_00"

#: Keys every segment carries from the moment ASR creates it.
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
    """One canonical segment. Later stages add keys, never rename these."""
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
    """Sort by time and renumber, so `id` always matches playback order."""
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
    """Attach a speaker to each ASR segment using time overlap.

    `turns` are diarization intervals: {"speaker_id", "start", "end"}. A segment
    takes the speaker it overlaps most. Whisper's boundaries do not line up with
    diarization boundaries, so a segment that overlaps nothing falls back to the
    nearest turn within `max_gap` seconds and otherwise keeps the default
    speaker. Segments are never split: an utterance where two people overlap is
    credited to whoever holds most of it, and that is recorded in
    `speaker_confidence` rather than hidden.
    """
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
    """Distinct speakers in first-appearance order."""
    seen: Dict[str, None] = {}
    for segment in segments:
        seen.setdefault(segment.get("speaker_id") or DEFAULT_SPEAKER, None)
    return list(seen)


def merge_turns(turns: Sequence[Dict], gap: float = 0.25) -> List[Dict]:
    """Join consecutive turns of one speaker separated by a very short pause.

    pyannote emits many short turns for one continuous utterance; merging them
    makes both the speaker assignment and the reference extraction steadier.
    """
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


def exclusive_regions(
    turns: Sequence[Dict],
    speaker: str,
    min_duration: float = 1.0,
) -> List[Dict]:
    """Regions where `speaker` talks and nobody else does.

    Voice cloning on a clip containing two voices produces a blend of them, so
    the reference cutter needs the parts of a speaker's turns that no other
    speaker overlaps.
    """
    mine = [turn for turn in turns if turn["speaker_id"] == speaker]
    others = [turn for turn in turns if turn["speaker_id"] != speaker]
    regions: List[Dict] = []
    for turn in mine:
        pieces = [(float(turn["start"]), float(turn["end"]))]
        for other in others:
            o_start, o_end = float(other["start"]), float(other["end"])
            trimmed = []
            for start, end in pieces:
                if o_end <= start or o_start >= end:
                    trimmed.append((start, end))
                    continue
                if o_start > start:
                    trimmed.append((start, o_start))
                if o_end < end:
                    trimmed.append((o_end, end))
            pieces = trimmed
        for start, end in pieces:
            if end - start >= min_duration:
                regions.append(
                    {"start": round(start, 3), "end": round(end, 3),
                     "duration": round(end - start, 3)}
                )
    regions.sort(key=lambda region: region["duration"], reverse=True)
    return regions


def text_between(segments: Sequence[Dict], start: float, end: float) -> str:
    """Source text spoken inside a window - the reference transcript some
    cloning engines want alongside the reference audio."""
    chosen = [
        segment["source_text"]
        for segment in sorted(segments, key=lambda item: item["start"])
        if _overlap(float(segment["start"]), float(segment["end"]), start, end)
        > 0.5 * max(1e-6, float(segment["end"]) - float(segment["start"]))
    ]
    return " ".join(part for part in chosen if part).strip()


def subtitle(segments: Sequence[Dict]) -> str:
    """SRT body, translated text when present and source text otherwise."""
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
    """Per-speaker counts for the job status payload."""
    summary: Dict[str, Dict] = {}
    for segment in segments:
        name = segment.get("speaker_id") or DEFAULT_SPEAKER
        entry = summary.setdefault(name, {"speaker_id": name, "segments": 0, "seconds": 0.0})
        entry["segments"] += 1
        entry["seconds"] = round(entry["seconds"] + float(segment.get("duration") or 0.0), 3)
    return list(summary.values())


def optional_float(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(float(value), 3)
