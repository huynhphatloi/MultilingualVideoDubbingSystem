"""VieNeu-TTS: Vietnamese cloning that also runs on CPU.

Voices are registered with the library once per reference clip and then reused,
so a job with three speakers pays the registration cost three times rather than
once per line.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Optional

from core.errors import ProviderFailure
from core.media import write_waveform
from providers.base import ModelSpec
from providers.tts.base import SpeechRequest, TTSProvider

SAMPLE_RATE_ENV = "VIENEU_SAMPLE_RATE"


class VieNeuProvider(TTSProvider):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__(spec)
        from vieneu import Vieneu

        self.model = Vieneu()
        self._voices: Dict[str, str] = {}

    @property
    def name(self) -> str:
        return self.spec.repo_id or "pnnbao-ump/VieNeu-TTS"

    def _voice_for(self, reference: Path) -> str:
        key = hashlib.sha1(str(reference.resolve()).encode("utf-8")).hexdigest()[:12]
        if key not in self._voices:
            self.model.add_voice(key, str(reference))
            self._voices[key] = key
        return self._voices[key]

    def synthesize(self, request: SpeechRequest, out: Path) -> None:
        import numpy

        reference = self.require_reference(request)
        audio = self.model.infer(request.text, self._voice_for(reference))
        array = numpy.asarray(audio, dtype="float32").reshape(-1)
        if array.size == 0:
            raise ProviderFailure("VieNeu-TTS produced no audio")
        rate = int(getattr(self.model, "sample_rate", 0) or 24000)
        write_waveform(array, rate, out, request.speed)


def cache_key(spec: ModelSpec, language: Optional[str] = None) -> str:
    return f"tts:{spec.id}"


def build(spec: ModelSpec, language: Optional[str] = None, **_: object) -> TTSProvider:
    return VieNeuProvider(spec)
