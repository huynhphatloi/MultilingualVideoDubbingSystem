"""What every translation engine has to provide."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence

from providers.base import ModelSpec


class TranslationProvider(ABC):
    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec

    @property
    def name(self) -> str:
        return self.spec.repo_id or self.spec.id

    @abstractmethod
    def translate(self, texts: Sequence[str], source: str, target: str) -> List[str]:
        pass
