"""Translation engine interface. Adding a new model = adding one subclass."""
from __future__ import annotations

import abc
from dataclasses import dataclass, field


@dataclass
class TranslationRequest:
    text: str
    source_language: str
    target_language: str
    #: Soft target for the number of characters in the output. When set the
    #: engine returns the candidate closest to this budget (duration-aware mode).
    char_budget: int | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class TranslationResult:
    text: str
    engine: str
    candidates: list[str] = field(default_factory=list)
    char_budget: int | None = None


class TranslationEngine(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def supports(self, source_language: str, target_language: str) -> bool:
        ...

    @abc.abstractmethod
    def translate_batch(self, requests: list[TranslationRequest]) -> list[TranslationResult]:
        ...

    def translate(self, request: TranslationRequest) -> TranslationResult:
        return self.translate_batch([request])[0]

    def warmup(self) -> None:  # pragma: no cover - optional
        return None
