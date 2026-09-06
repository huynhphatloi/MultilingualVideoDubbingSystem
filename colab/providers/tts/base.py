"""Interfaces shared by speech generation providers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from providers.base import ModelSpec


@dataclass
class SpeechRequest:
    text: str
    language: str
    speed: float = 1.0
    speaker_id: Optional[str] = None
    multi_voice: bool = False


class TTSProvider(ABC):
    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec

    @property
    def name(self) -> str:
        return self.spec.repo_id or self.spec.id

    @property
    def selected_voice(self) -> Optional[str]:
        return None

    @abstractmethod
    def synthesize(self, request: SpeechRequest, out: Path) -> None:
        pass
