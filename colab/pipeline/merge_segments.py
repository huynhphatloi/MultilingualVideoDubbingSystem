"""Give every recognised segment a speaker.

With diarization the speaker is whoever holds most of the segment; without it
everything belongs to the single default speaker. Either way the segment schema
is identical from here on.
"""
from __future__ import annotations

import json
from pathlib import Path

from dubflow_core.segments import (
    assign_speakers,
    public_speaker_summary,
    renumber,
    speakers_of,
)

NAME = "merge_segments"


def run(job: dict, folder: Path) -> None:
    segments = renumber(assign_speakers(job.get("segments", []), job.get("turns") or []))
    job["segments"] = segments
    job["speakers"] = speakers_of(segments)
    job["speaker_summary"] = public_speaker_summary(segments)
    (folder / "transcript.json").write_text(
        json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8"
    )
