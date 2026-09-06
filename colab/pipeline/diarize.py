"""Speaker diarization stage."""
from __future__ import annotations

import json
from pathlib import Path

import providers
from core.config import rebuild
from dubflow_core.segments import speakers_of

NAME = "diarize"


def enabled(job: dict) -> bool:
    return bool(job.get("config", {}).get("features", {}).get("diarization"))


def run(job: dict, folder: Path) -> None:
    config = rebuild(job)
    provider = providers.diarization.load(config.diarization)
    turns = provider.diarize(
        folder / job["files"]["asr_audio"],
        min_speakers=config.min_speakers,
        max_speakers=config.max_speakers,
    )
    job["turns"] = turns
    job["speakers"] = speakers_of(turns)
    job["diarization_model"] = provider.name
    (folder / "diarization.json").write_text(
        json.dumps(turns, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    job["files"]["diarization"] = "diarization.json"
