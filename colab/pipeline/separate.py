"""Split the original track so the dub can sit on the real background.

Optional and off by default: it is the slowest stage in the pipeline and the
voice-over mix works without it.
"""
from __future__ import annotations

from pathlib import Path

import providers
from core.config import rebuild

NAME = "separate"


def enabled(job: dict) -> bool:
    return bool(job.get("config", {}).get("features", {}).get("source_separation"))


def run(job: dict, folder: Path) -> None:
    config = rebuild(job)
    provider = providers.separation.load(config.separation)
    stems = provider.separate(folder / job["files"]["original_audio"], folder)
    job["files"]["background_audio"] = str(stems["background"].relative_to(folder))
    job["files"]["original_speech"] = str(stems["speech"].relative_to(folder))
    job["separation_model"] = provider.name
