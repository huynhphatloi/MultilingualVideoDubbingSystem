"""Resemble AI Chatterbox.

Two checkpoints share the package: the multilingual one, which takes a language
id, and the English-only one, which does not. Both clone from a reference clip
and both fall back to their built-in voice when none is given, so the reference
is optional rather than required.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from core.media import write_waveform
from core.runtime import device
from providers.base import ModelSpec
from providers.tts.base import SpeechRequest, TTSProvider


class ChatterboxProvider(TTSProvider):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__(spec)
        self.multilingual = spec.id != "chatterbox_en"
        if self.multilingual:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS as Model
        else:
            from chatterbox.tts import ChatterboxTTS as Model

        self.model = Model.from_pretrained(device=device())

    @property
    def name(self) -> str:
        return f"chatterbox/{'multilingual' if self.multilingual else 'english'}"

    def synthesize(self, request: SpeechRequest, out: Path) -> None:
        arguments = {}
        if request.reference is not None and Path(request.reference).exists():
            arguments["audio_prompt_path"] = str(request.reference)
        if self.multilingual:
            waveform = self.model.generate(
                request.text, language_id=request.language, **arguments
            )
        else:
            waveform = self.model.generate(request.text, **arguments)
        audio = waveform.squeeze(0).detach().cpu().numpy()
        write_waveform(audio, int(self.model.sr), out, request.speed)


def cache_key(spec: ModelSpec, language: Optional[str] = None) -> str:
    return f"tts:{spec.id}"


def build(spec: ModelSpec, language: Optional[str] = None, **_: object) -> TTSProvider:
    return ChatterboxProvider(spec)
