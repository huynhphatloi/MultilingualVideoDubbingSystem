"""Speech to text.

A recogniser with its own timestamps transcribes the whole track. One without
them is given windows instead: the diarization turns when that stage ran, and
voice-activity windows otherwise. Either way the stage produces the same
canonical segments.
"""
from __future__ import annotations

import json
from pathlib import Path

import providers
from core.config import rebuild
from core.errors import ProviderFailure
from providers.asr import windows_from_turns

NAME = "transcribe"


def run(job: dict, folder: Path) -> None:
    config = rebuild(job)
    audio = folder / job["files"]["asr_audio"]
    provider = providers.asr.load(config.asr, config.source_language)

    # `None` means "detect it". A recogniser with its own detection reports the
    # language as part of transcribing, and a windowed one calls its own
    # detect_language before cutting - neither needs a separate pass here.
    language = config.source_language

    windows = None
    if not config.asr.supports_timestamps and job.get("turns"):
        # The turns already say where speech is and who owns it, so a windowed
        # recogniser gets its segmentation for free and keeps the speaker label.
        windows = windows_from_turns(job["turns"])

    transcript = provider.transcribe(audio, language, windows)
    if not transcript.segments:
        raise ProviderFailure(
            f"'{config.asr.display_name}' found no speech in the video"
        )

    config.confirm_source_language(transcript.language)
    if config.source_language is None:
        job["detected_language"] = transcript.language
    job["source_language"] = transcript.language
    job["segments"] = transcript.segments
    job["asr_model"] = provider.name
    (folder / "transcript.json").write_text(
        json.dumps(transcript.segments, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    job["files"]["transcript"] = "transcript.json"
