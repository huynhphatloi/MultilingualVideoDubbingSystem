"""SpeechGenerator abstraction.

    SpeechGenerator
     |
     +-- XTTSAdapter          (multilingual voice cloning, 17 languages)
     +-- MMSAdapter           (single-speaker, ~1100 languages - the safety net)
     +-- F5Adapter            (optional, high quality cloning)

Adapters never translate. They receive already-translated text plus an optional
voice reference and return a wav file.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SynthesisRequest:
    text: str
    language: str
    output_path: Path
    voice_reference: Path | None = None
    speaker_id: str | None = None
    speed: float = 1.0
    metadata: dict = field(default_factory=dict)


@dataclass
class SynthesisResult:
    path: Path
    duration: float
    model: str
    voice_cloned: bool
    sample_rate: int
    #: True when the engine already cast THIS speaker differently from the
    #: others (Edge-TTS picks a woman for SPEAKER_00 and a man for
    #: SPEAKER_01). Distinct from `voice_cloned`, which means "this is the
    #: original actor". The synthesis stage pitch-shifts speakers apart only
    #: when neither holds - shifting a voice that already differs makes the
    #: male read unnaturally deep for no gain.
    distinct_voice: bool = False


class SpeechGenerator(abc.ABC):
    #: Stable id. Also the `force_model` key, so it must not drift.
    name: str = "base"
    #: Human label for the model picker in the UI and for reports.
    title: str = "Base"
    #: One line on what choosing this engine costs or buys. The picker shows
    #: it under the dropdown, because "vixtts vs f5_vi" means nothing to
    #: someone who has not read the model cards.
    blurb: str = ""
    supports_cloning: bool = False

    @abc.abstractmethod
    def supports(self, language: str) -> bool:
        """Can this adapter speak the language at all?"""

    @abc.abstractmethod
    def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        ...

    def unavailable_reason(self, language: str) -> str:
        """Why supports() said no, in words an operator can act on.

        The default assumes the language is the problem. Adapters that can be
        switched off or misconfigured MUST override it: "F5-TTS does not speak
        'en'" is simply false - it does, it is just disabled - and a picker
        that says so sends the reader looking for the wrong fix.
        """
        return f"does not speak '{language}'"

    def describe(self) -> dict:
        return {"name": self.name, "title": self.title, "blurb": self.blurb,
                "voice_cloning": self.supports_cloning}
