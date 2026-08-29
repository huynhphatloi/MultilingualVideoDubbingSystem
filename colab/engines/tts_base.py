"""Contract every TTS engine on the Colab side implements.

Deliberately the same shape as ``ai-service/app/services/tts/base.py`` so the
two sides stay readable together: text in, wav out, plus enough metadata for
the client to know *who* spoke and whether the voice was actually cloned.

Adding an engine is: subclass, fill in three methods, register it in
``engines/__init__.py``. Nothing else in the system needs to change - the
client discovers engines over ``GET /health``.
"""
from __future__ import annotations

import abc
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Spoken:
    """One synthesis result."""

    wav_path: Path
    sample_rate: int
    seconds: float
    #: Wall-clock seconds spent inside the model. Divided by `seconds` this is
    #: the real-time factor, the only speed number worth comparing.
    compute_seconds: float
    voice_cloned: bool
    engine: str
    notes: dict = field(default_factory=dict)

    @property
    def rtf(self) -> float:
        return self.compute_seconds / self.seconds if self.seconds else 0.0


class TTSEngine(abc.ABC):
    """A speech synthesiser hosted on the GPU box."""

    #: Short, stable id used on the wire (``X-TTS-Engine``) and in reports.
    name: str = "base"
    #: Human label for the comparison table.
    title: str = "Base"
    #: True when the engine reproduces the reference speaker's voice.
    voice_cloning: bool = False
    #: ISO-639-1 codes. "*" means "anything the checkpoint has", which is how
    #: MMS (1100+) and Piper (50+) declare themselves without listing them all.
    languages: tuple[str, ...] = ()
    #: Roughly how much VRAM the loaded model needs, for the notebook to warn
    #: before it tries to hold four of them at once on a 16 GB T4.
    vram_gb: float = 0.0

    def __init__(self) -> None:
        self._loaded = False
        # One GPU, one model at a time. Every engine shares this discipline so
        # concurrent requests queue instead of racing for VRAM.
        self._lock = threading.Lock()

    # ------------------------------------------------------------ lifecycle --
    @abc.abstractmethod
    def load(self) -> None:
        """Bring the weights into memory. Called once, lazily."""

    def ensure_loaded(self) -> None:
        if not self._loaded:
            with self._lock:
                if not self._loaded:
                    started = time.perf_counter()
                    self.load()
                    self._loaded = True
                    self.load_seconds = round(time.perf_counter() - started, 1)

    def unload(self) -> None:
        """Free VRAM. Default is a no-op; heavy engines should override."""
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------ capability --
    def supports(self, language: str) -> bool:
        return "*" in self.languages or language in self.languages

    def needs_reference(self) -> bool:
        return self.voice_cloning

    # -------------------------------------------------------------- speaking --
    @abc.abstractmethod
    def speak(self, text: str, language: str, out_path: Path,
              reference_wav: Path | None = None, speed: float = 1.0) -> Spoken:
        """Synthesise ``text`` into ``out_path``."""

    # ------------------------------------------------------------ describing --
    def describe(self) -> dict:
        return {
            "name": self.name,
            "title": self.title,
            "voice_cloning": self.voice_cloning,
            "languages": list(self.languages),
            "vram_gb": self.vram_gb,
            "loaded": self._loaded,
            "load_seconds": getattr(self, "load_seconds", None),
        }
