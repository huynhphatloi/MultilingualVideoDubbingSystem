"""Coqui XTTS v2 and the viXTTS Vietnamese fine-tune.

Both come from the coqui-tts package but load differently: XTTS through the
high-level API, viXTTS from a checkpoint directory. One module, two branches, so
the coqui import lives in one place.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from core.media import Scratch, normalise_speech, write_waveform
from core.runtime import device
from providers.base import ModelSpec
from providers.tts.base import SpeechRequest, TTSProvider

#: XTTS expects "zh-cn" where this project says "zh".
_LANGUAGE_OVERRIDES = {"zh": "zh-cn"}


class XTTSProvider(TTSProvider):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__(spec)
        from TTS.api import TTS

        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device())

    @property
    def name(self) -> str:
        return "coqui/xtts_v2"

    def synthesize(self, request: SpeechRequest, out: Path) -> None:
        sample = self.require_reference(request)
        with Scratch(".wav") as raw:
            self.tts.tts_to_file(
                text=request.text,
                speaker_wav=str(sample),
                language=_LANGUAGE_OVERRIDES.get(request.language, request.language),
                file_path=str(raw),
            )
            normalise_speech(raw, out, request.speed)


class ViXTTSProvider(TTSProvider):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__(spec)
        from huggingface_hub import snapshot_download
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

        folder = snapshot_download(spec.repo_id or "capleaf/viXTTS")
        self.config = XttsConfig()
        self.config.load_json(str(Path(folder) / "config.json"))
        model = Xtts.init_from_config(self.config)
        model.load_checkpoint(self.config, checkpoint_dir=folder, use_deepspeed=False)
        self.model = model.to(device())

    @property
    def name(self) -> str:
        return self.spec.repo_id or "capleaf/viXTTS"

    def synthesize(self, request: SpeechRequest, out: Path) -> None:
        import torch

        sample = self.require_reference(request)
        with torch.inference_mode():
            result = self.model.synthesize(
                request.text, self.config, speaker_wav=str(sample), language="vi"
            )
        write_waveform(result["wav"], 24000, out, request.speed)


def cache_key(spec: ModelSpec, language: Optional[str] = None) -> str:
    return f"tts:{spec.id}"


def build(spec: ModelSpec, language: Optional[str] = None, **_: object) -> TTSProvider:
    return ViXTTSProvider(spec) if spec.id == "vixtts" else XTTSProvider(spec)
