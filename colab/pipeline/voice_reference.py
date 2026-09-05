"""Cut one clean reference clip per speaker.

The old pipeline cut a single twelve-second clip from the start of the video and
cloned every line from it, which in a two-person interview means both speakers
end up sounding like whoever talked first. With diarization this cuts one clip
per speaker from regions nobody else overlaps; without it there is one speaker,
so the behaviour is the same as before.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from core.errors import ProviderFailure
from core.media import Scratch, concatenate, extract_window
from dubflow_core.segments import DEFAULT_SPEAKER, exclusive_regions, text_between

NAME = "voice_references"

#: Cloning quality plateaus well before this; longer clips only cost time.
TARGET_SECONDS = 12.0
#: Below this the clone is unstable, so the stage says so instead of trying.
MINIMUM_SECONDS = 1.0
#: Never take more than this from one region, so the clip mixes a few phrases.
MAX_PIECE_SECONDS = 8.0


def enabled(job: dict) -> bool:
    return bool(job.get("config", {}).get("features", {}).get("voice_cloning"))


def _regions_for(job: dict, speaker: str) -> List[Dict]:
    turns = job.get("turns") or []
    if turns:
        regions = exclusive_regions(turns, speaker, min_duration=MINIMUM_SECONDS)
        if regions:
            return regions
    # No diarization, or a speaker whose every turn overlaps someone else: fall
    # back to that speaker's own recognised segments.
    return [
        {"start": float(segment["start"]), "end": float(segment["end"]),
         "duration": float(segment["end"]) - float(segment["start"])}
        for segment in sorted(
            (item for item in job["segments"]
             if (item.get("speaker_id") or DEFAULT_SPEAKER) == speaker),
            key=lambda item: float(item["end"]) - float(item["start"]),
            reverse=True,
        )
    ]


def run(job: dict, folder: Path) -> None:
    segments = job["segments"]
    if not segments:
        raise ProviderFailure("A voice reference needs the transcript, which is empty")

    original = folder / job["files"]["original_audio"]
    output_dir = folder / "references"
    output_dir.mkdir(exist_ok=True)

    speakers = job.get("speakers") or [DEFAULT_SPEAKER]
    references: Dict[str, dict] = {}
    for speaker in speakers:
        regions = _regions_for(job, speaker)
        chosen, total = [], 0.0
        for region in regions:
            if total >= TARGET_SECONDS:
                break
            length = min(region["duration"], MAX_PIECE_SECONDS, TARGET_SECONDS - total)
            if length < 0.4:
                continue
            chosen.append({"start": region["start"], "length": round(length, 3)})
            total += length
        if total < MINIMUM_SECONDS:
            raise ProviderFailure(
                f"{speaker} has only {total:.2f}s of clean speech, too little to "
                f"clone a voice. Run the job with voice cloning off, or use a "
                f"clip where each speaker talks for at least a second."
            )

        target = output_dir / f"{speaker}.wav"
        pieces: List[Path] = []
        scratches = [Scratch(".wav") for _ in chosen]
        try:
            for scratch, piece in zip(scratches, chosen):
                path = scratch.__enter__()
                extract_window(original, path, piece["start"], piece["length"])
                pieces.append(path)
            concatenate(pieces, target)
        finally:
            for scratch in scratches:
                scratch.__exit__()

        spoken = " ".join(
            text_between(segments, piece["start"], piece["start"] + piece["length"])
            for piece in chosen
        ).strip()
        references[speaker] = {
            "file": str(target.relative_to(folder)),
            "seconds": round(total, 3),
            "text": spoken,
            "exclusive": bool(job.get("turns")),
        }

    job["references"] = references
    job["files"]["references"] = "references"
