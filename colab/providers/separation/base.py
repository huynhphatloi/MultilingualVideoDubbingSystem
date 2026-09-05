"""Source separation: splitting the original track into speech and background."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict

from providers.base import ModelSpec


class SourceSeparationProvider(ABC):
    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec

    @property
    def name(self) -> str:
        return self.spec.repo_id or self.spec.id

    @abstractmethod
    def separate(self, audio: Path, workdir: Path) -> Dict[str, Path]:
        """Return {"speech": ..., "background": ...} as WAV files in workdir."""
