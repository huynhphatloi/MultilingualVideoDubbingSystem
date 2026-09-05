"""Lip sync: re-timing the mouth in the video to the dubbed audio."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from providers.base import ModelSpec


class LipSyncProvider(ABC):
    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec

    @property
    def name(self) -> str:
        return self.spec.repo_id or self.spec.id

    @abstractmethod
    def synchronise(self, video: Path, audio: Path, out: Path) -> None:
        """Write a video whose mouth movement follows `audio`."""
