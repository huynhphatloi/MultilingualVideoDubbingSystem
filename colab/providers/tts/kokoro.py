"""Kokoro 82M: stock voices, Apache-2.0, no cloning.

The pipeline is per language code, and the voice name has to match that code's
first letter, so both are chosen here from the model's published voice list.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from core.errors import ProviderFailure, UnsupportedLanguage
from core.media import write_waveform
from core.runtime import device
from providers.base import ModelSpec
from providers.tts.base import SpeechRequest, TTSProvider

#: Application code -> (Kokoro pipeline code, default voice). American English
#: and Brazilian Portuguese are the defaults for their languages; the voices are
#: taken from the checkpoint's own voices/ directory.
PIPELINES: Dict[str, tuple] = {
    "en": ("a", "af_heart"),
    "es": ("e", "ef_dora"),
    "fr": ("f", "ff_siwis"),
    "hi": ("h", "hf_alpha"),
    "it": ("i", "if_sara"),
    "ja": ("j", "jf_alpha"),
    "pt": ("p", "pf_dora"),
    "zh": ("z", "zf_xiaoxiao"),
}
SAMPLE_RATE = 24000


class KokoroProvider(TTSProvider):
    def __init__(self, spec: ModelSpec, language: str) -> None:
        super().__init__(spec)
        from kokoro import KPipeline

        if language not in PIPELINES:
            raise UnsupportedLanguage(f"Kokoro has no pipeline for '{language}'.")
        self.code, self.voice = PIPELINES[language]
        self.pipeline = KPipeline(
            lang_code=self.code,
            repo_id=spec.repo_id or "hexgrad/Kokoro-82M",
            device=device(),
        )

    @property
    def name(self) -> str:
        return f"kokoro/{self.voice}"

    def synthesize(self, request: SpeechRequest, out: Path) -> None:
        import numpy

        pieces = []
        for result in self.pipeline(request.text, voice=self.voice, speed=request.speed):
            audio = getattr(result, "audio", None)
            if audio is None and isinstance(result, (tuple, list)):
                audio = result[-1]
            if audio is not None:
                pieces.append(numpy.asarray(audio, dtype="float32").reshape(-1))
        if not pieces:
            raise ProviderFailure("Kokoro produced no audio")
        # Kokoro splits long text into chunks; the caller wants one clip.
        write_waveform(numpy.concatenate(pieces), SAMPLE_RATE, out)


def cache_key(spec: ModelSpec, language: Optional[str] = None) -> str:
    return f"tts:{spec.id}:{language}"


def build(spec: ModelSpec, language: str, **_: object) -> TTSProvider:
    return KokoroProvider(spec, language)
