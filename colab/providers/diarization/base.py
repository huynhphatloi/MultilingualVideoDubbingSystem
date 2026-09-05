"""Speaker diarization.

Diarization answers "who spoke when", not "who is speaking": the labels are
SPEAKER_00, SPEAKER_01 and so on, and nothing in this project pretends they are
names.
"""
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
        """Turns as {"speaker_id", "start", "end"}, in time order."""
