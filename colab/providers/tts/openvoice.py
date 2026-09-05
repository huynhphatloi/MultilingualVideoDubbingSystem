"""OpenVoice V2, which is MeloTTS plus a tone-colour converter.

MeloTTS speaks the line in a stock voice; OpenVoice then moves that recording
onto the reference speaker's timbre. Neither part is on PyPI, so both are
installed from git and the registry marks the model experimental.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

from core.errors import MissingDependency, ProviderFailure, UnsupportedLanguage
from core.media import Scratch, normalise_speech
from core.runtime import device
from providers.base import ModelSpec
from providers.tts.base import SpeechRequest, TTSProvider

#: Application code -> the MeloTTS language name.
MELO_LANGUAGES: Dict[str, str] = {
    "en": "EN", "es": "ES", "fr": "FR", "zh": "ZH", "ja": "JP", "ko": "KR",
}


class OpenVoiceProvider(TTSProvider):
    def __init__(self, spec: ModelSpec, language: str) -> None:
        super().__init__(spec)
        if language not in MELO_LANGUAGES:
            raise UnsupportedLanguage(f"OpenVoice V2 has no base speaker for '{language}'.")
        try:
            from melo.api import TTS as MeloTTS
        except ImportError as exc:
            raise MissingDependency(
                "git+https://github.com/myshell-ai/MeloTTS.git",
                "OpenVoice V2 (its base speaker)",
            ) from exc
        from huggingface_hub import snapshot_download
        from openvoice.api import ToneColorConverter

        self.language = language
        self.checkpoints = Path(
            os.getenv("OPENVOICE_CHECKPOINTS")
            or snapshot_download(spec.repo_id or "myshell-ai/OpenVoiceV2")
        )
        converter = self.checkpoints / "converter"
        self.converter = ToneColorConverter(str(converter / "config.json"), device=device())
        self.converter.load_ckpt(str(converter / "checkpoint.pth"))

        self.melo = MeloTTS(language=MELO_LANGUAGES[language], device=device())
        speakers = self.melo.hps.data.spk2id
        self.speaker_name = list(speakers.keys())[0]
        self.speaker_id = speakers[self.speaker_name]
        # The demo derives the embedding filename from the speaker key.
        key = self.speaker_name.lower().replace("_", "-")
        self.source_embedding = self.checkpoints / "base_speakers" / "ses" / f"{key}.pth"
        if not self.source_embedding.exists():
            raise ProviderFailure(
                f"OpenVoice has no speaker embedding for '{key}' in {self.checkpoints}"
            )
        self._target_cache: Dict[str, object] = {}

    @property
    def name(self) -> str:
        return f"openvoice_v2/{MELO_LANGUAGES[self.language]}-{self.speaker_name}"

    def _target_embedding(self, reference: Path):  # noqa: ANN202
        from openvoice import se_extractor

        key = str(reference)
        if key not in self._target_cache:
            embedding, _ = se_extractor.get_se(str(reference), self.converter, vad=True)
            self._target_cache[key] = embedding
        return self._target_cache[key]

    def synthesize(self, request: SpeechRequest, out: Path) -> None:
        import torch

        reference = self.require_reference(request)
        source = torch.load(str(self.source_embedding), map_location=device())
        target = self._target_embedding(reference)
        with Scratch(".wav") as spoken, Scratch(".wav") as converted:
            self.melo.tts_to_file(
                request.text, self.speaker_id, str(spoken), speed=request.speed
            )
            self.converter.convert(
                audio_src_path=str(spoken),
                src_se=source,
                tgt_se=target,
                output_path=str(converted),
            )
            normalise_speech(converted, out)


def cache_key(spec: ModelSpec, language: Optional[str] = None) -> str:
    return f"tts:{spec.id}:{language}"


def build(spec: ModelSpec, language: str, **_: object) -> TTSProvider:
    return OpenVoiceProvider(spec, language)
