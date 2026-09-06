"""Speaker diarization provider interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

from providers.base import ModelSpec


class DiarizationProvider(ABC):
    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec

    @property
    def name(self) -> str:
        return self.spec.repo_id or self.spec.id

    @abstractmethod
    def diarize(
        self,
        audio: Path,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> List[Dict]:
        pass
