"""Optional lip sync.

Off by default, and this build registers no provider - see
providers/lipsync/registry.py for why and for what adding one involves. The
stage and the flag exist so that adding a provider is the only work required.
"""
from __future__ import annotations

from pathlib import Path

import providers
from core.config import rebuild

NAME = "lipsync"


def enabled(job: dict) -> bool:
    return bool(job.get("config", {}).get("features", {}).get("lip_sync"))


def run(job: dict, folder: Path) -> None:
    config = rebuild(job)
    provider = providers.lipsync.load(config.lipsync)
    output = folder / "lipsynced.mp4"
    provider.synchronise(
        folder / job["files"]["input"],
        folder / job["files"]["final_audio"],
        output,
    )
    job["files"]["lipsync_video"] = output.name
    job["lipsync_model"] = provider.name
