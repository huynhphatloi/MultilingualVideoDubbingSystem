"""F5-TTS: the v1 base checkpoint and the Vietnamese community one.

F5 wants the transcript of the reference clip. Passing an empty string makes it
transcribe the clip itself, which is what this project used to do; now that the
pipeline knows what the reference speaker said, it passes the text and saves
that pass.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from core.media import Scratch, normalise_speech
from providers.base import ModelSpec
from providers.tts.base import SpeechRequest, TTSProvider

#: Checkpoint names as f5_tts.api.F5TTS understands them.
_CHECKPOINTS = {"f5_base": "F5TTS_v1_Base", "f5_vi": None}
#: Kept so an existing .env that pins the Vietnamese checkpoint still wins.
VI_MODEL_ENV = "F5_VI_MODEL"


def checkpoint_for(spec: ModelSpec) -> str:
    if spec.id == "f5_vi":
        return os.getenv(VI_MODEL_ENV, spec.repo_id or "hynt/F5-TTS-Vietnamese-ViVoice")
    return _CHECKPOINTS.get(spec.id) or "F5TTS_v1_Base"


class F5Provider(TTSProvider):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__(spec)
        from f5_tts.api import F5TTS

        self.checkpoint = checkpoint_for(spec)
        self.model = F5TTS(model=self.checkpoint)

    @property
    def name(self) -> str:
        return f"f5-tts/{self.checkpoint}"

    def synthesize(self, request: SpeechRequest, out: Path) -> None:
        sample = self.require_reference(request)
        with Scratch(".wav") as raw:
            self.model.infer(
                ref_file=str(sample),
                # An empty ref_text makes F5 transcribe the reference itself.
                ref_text=(request.reference_text or "").strip(),
                gen_text=request.text,
                file_wave=str(raw),
            )
            normalise_speech(raw, out, request.speed)


def cache_key(spec: ModelSpec, language: Optional[str] = None) -> str:
    return f"tts:{spec.id}"


def build(spec: ModelSpec, language: Optional[str] = None, **_: object) -> TTSProvider:
    return F5Provider(spec)
