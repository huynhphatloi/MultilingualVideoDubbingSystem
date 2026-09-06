"""Interfaces shared by speech generation providers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from core.errors import InvalidRequest
from providers.base import ModelSpec


@dataclass
class SpeechRequest:
    text: str
    language: str
    speed: float = 1.0
    reference: Optional[Path] = None
    reference_text: Optional[str] = None
    speaker_id: Optional[str] = None


class TTSProvider(ABC):
    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec

    @property
    def name(self) -> str:
        return self.spec.repo_id or self.spec.id

    def require_reference(self, request: SpeechRequest) -> Path:
        if request.reference is None or not Path(request.reference).exists():
            raise InvalidRequest(
                f"'{self.spec.display_name}' clones a voice and needs a reference "
                f"sample. Upload one to POST /reference and pass its reference_id, "
                f"or run the job with diarization and voice cloning enabled so the "
                f"pipeline cuts one per speaker."
            )
        return Path(request.reference)

    def require_reference_text(self, request: SpeechRequest) -> str:
        text = (request.reference_text or "").strip()
        if not text:
            raise InvalidRequest(
                f"'{self.spec.display_name}' needs the transcript of the reference "
                f"clip as well as the audio. The pipeline supplies it from the "
                f"recognised text; a direct /synthesize call must pass "
                f"reference_text."
            )
        return text

    @abstractmethod
    def synthesize(self, request: SpeechRequest, out: Path) -> None:
        pass
