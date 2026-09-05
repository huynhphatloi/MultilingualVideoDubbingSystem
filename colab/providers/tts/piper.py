"""Piper: small ONNX voices that run on CPU."""
from __future__ import annotations

import wave
from pathlib import Path
from typing import Optional

from core.errors import UnsupportedLanguage
from core.media import Scratch, normalise_speech
from providers.base import ModelSpec
from providers.tts.base import SpeechRequest, TTSProvider


class PiperProvider(TTSProvider):
    def __init__(self, spec: ModelSpec, language: str) -> None:
        super().__init__(spec)
        from huggingface_hub import hf_hub_download, list_repo_files
        from piper.voice import PiperVoice

        repo = spec.repo_id or "rhasspy/piper-voices"
        candidates = [
            name for name in list_repo_files(repo)
            if name.startswith(f"{language}/") and name.endswith(".onnx")
        ]
        if not candidates:
            raise UnsupportedLanguage(f"Piper publishes no voice for '{language}'.")
        # Prefer the medium build: low is noticeably rougher, high much slower.
        candidates.sort(key=lambda name: (0 if "/medium/" in name else 1, name))
        self.voice_name = candidates[0]
        onnx = hf_hub_download(repo, self.voice_name)
        config = hf_hub_download(repo, f"{self.voice_name}.json")
        self.voice = PiperVoice.load(onnx, config_path=config)

    @property
    def name(self) -> str:
        return f"piper/{self.voice_name}"

    def synthesize(self, request: SpeechRequest, out: Path) -> None:
        with Scratch(".wav") as raw:
            with wave.open(str(raw), "wb") as handle:
                self.voice.synthesize(request.text, handle)
            normalise_speech(raw, out, request.speed)


def cache_key(spec: ModelSpec, language: Optional[str] = None) -> str:
    return f"tts:{spec.id}:{language}"


def build(spec: ModelSpec, language: str, **_: object) -> TTSProvider:
    return PiperProvider(spec, language)
