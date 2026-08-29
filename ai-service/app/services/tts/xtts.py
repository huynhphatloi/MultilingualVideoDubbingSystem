"""XTTS-v2 adapter (Coqui) - multilingual zero-shot voice cloning.

Free, runs locally, and covers 17 languages. This is the preferred engine
whenever the target language is on its list and we have a voice reference.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from app.core import languages
from app.core.config import settings
from app.core.device import resolve_device_for
from app.core.errors import ModelLoadError, SynthesisError
from app.services import ffmpeg
from app.services.models_registry import get_or_load
from app.services.tts.base import SpeechGenerator, SynthesisRequest, SynthesisResult

log = logging.getLogger(__name__)


class XTTSAdapter(SpeechGenerator):
    name = "xtts_v2"
    title = "XTTS-v2 (local, clones)"
    blurb = ("17 languages, clones each speaker from a few seconds of their "
             "voice. Vietnamese is not one of the 17.")
    supports_cloning = True

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.xtts_model

    def _model(self):  # noqa: ANN201
        def loader():  # noqa: ANN202
            os.environ.setdefault("COQUI_TOS_AGREED", "1")
            try:
                from TTS.api import TTS
            except ImportError as exc:  # pragma: no cover
                raise ModelLoadError(
                    "coqui-tts is not installed (pip install coqui-tts)"
                ) from exc
            try:
                return TTS(self.model_name).to(resolve_device_for("tts"))
            except Exception as exc:
                raise ModelLoadError(f"Cannot load XTTS: {exc}") from exc

        return get_or_load(f"xtts::{self.model_name}", loader)

    def supports(self, language: str) -> bool:
        return languages.to_xtts(language) is not None

    def unavailable_reason(self, language: str) -> str:
        return f"XTTS-v2's 17 languages do not include '{language}'"

    def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        lang = languages.to_xtts(request.language)
        if lang is None:
            raise SynthesisError(f"XTTS does not support '{request.language}'")
        if request.voice_reference is None or not Path(request.voice_reference).exists():
            raise SynthesisError("XTTS requires a speaker reference wav")

        model = self._model()
        out = Path(request.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        try:
            model.tts_to_file(
                text=request.text,
                speaker_wav=str(request.voice_reference),
                language=lang,
                file_path=str(out),
                speed=request.speed,
                split_sentences=True,
            )
        except Exception as exc:
            raise SynthesisError(
                f"XTTS synthesis failed: {exc}",
                details={"language": lang, "chars": len(request.text)},
            ) from exc

        if not out.exists() or out.stat().st_size == 0:
            raise SynthesisError("XTTS produced an empty file")

        return SynthesisResult(
            path=out, duration=ffmpeg.duration_of(out), model=self.name,
            voice_cloned=True, sample_rate=24000,
        )
