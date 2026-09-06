"""Meta MMS-TTS speech generation."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from core.errors import ProviderFailure, UnsupportedLanguage
from core.media import write_waveform
from core.runtime import device
from dubflow_core import languages as L
from providers.base import ModelSpec
from providers.tts.base import SpeechRequest, TTSProvider
from providers.tts.registry import MMS_TTS_LANGUAGES, MMS_TTS_OVERRIDES


def repository_for(language: str) -> str:
    if language not in MMS_TTS_LANGUAGES:
        raise UnsupportedLanguage(
            f"MMS-TTS publishes no checkpoint for '{L.name_of(language)}' "
            f"({language}). Pick another engine for this language."
        )
    return f"facebook/mms-tts-{MMS_TTS_OVERRIDES.get(language, L.iso3(language))}"


class MMSTTSProvider(TTSProvider):
    def __init__(self, spec: ModelSpec, language: str) -> None:
        super().__init__(spec)
        from transformers import AutoTokenizer, VitsModel

        self.language = language
        self.repo = repository_for(language)
        self.tokenizer = AutoTokenizer.from_pretrained(self.repo)
        self.model = VitsModel.from_pretrained(self.repo).to(device()).eval()

    @property
    def name(self) -> str:
        return self.repo

    def synthesize(self, request: SpeechRequest, out: Path) -> None:
        import torch

        encoded = self.tokenizer(request.text, return_tensors="pt")
        encoded = {name: value.to(device()) for name, value in encoded.items()}
        self.model.speaking_rate = request.speed
        with torch.inference_mode():
            waveform = self.model(**encoded).waveform
        audio = waveform.squeeze().float().cpu().numpy()
        if audio.size == 0:
            raise ProviderFailure("MMS-TTS produced no audio")
        write_waveform(audio, int(self.model.config.sampling_rate), out)


def cache_key(spec: ModelSpec, language: Optional[str] = None) -> str:
    return f"tts:{spec.id}:{language}"


def build(spec: ModelSpec, language: str, **_: object) -> TTSProvider:
    return MMSTTSProvider(spec, language)
