"""F5-TTS adapter - optional high quality cloning engine.

Disabled by default (``F5_TTS_ENABLED=false``) because the checkpoints are
large; enabling it only changes which adapter the router picks first.
"""
from __future__ import annotations

import logging
from pathlib import Path

from app.core import languages
from app.core.config import settings
from app.core.device import resolve_device_for
from app.core.errors import ModelLoadError, SynthesisError
from app.services import ffmpeg
from app.services.models_registry import get_or_load
from app.services.tts.base import SpeechGenerator, SynthesisRequest, SynthesisResult

log = logging.getLogger(__name__)

#: Officially released F5-TTS checkpoints.
_SUPPORTED = {"en", "zh"}


class F5Adapter(SpeechGenerator):
    name = "f5_tts"
    title = "F5-TTS (local, clones)"
    blurb = ("English and Chinese only. Off unless F5_TTS_ENABLED=true, and "
             "slow here - it has no Metal kernels, so it runs on the CPU.")
    supports_cloning = True

    def _model(self):  # noqa: ANN201
        def loader():  # noqa: ANN202
            try:
                from f5_tts.api import F5TTS
            except ImportError as exc:  # pragma: no cover
                raise ModelLoadError("f5-tts is not installed") from exc
            return F5TTS(device=resolve_device_for("tts"))

        return get_or_load("f5tts::default", loader)

    def supports(self, language: str) -> bool:
        if not settings.f5_tts_enabled:
            return False
        return (languages.normalize(language) or "") in _SUPPORTED

    def unavailable_reason(self, language: str) -> str:
        if not settings.f5_tts_enabled:
            return "disabled — set F5_TTS_ENABLED=true"
        return f"no released checkpoint for '{language}' (English and Chinese only)"

    def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        if not settings.f5_tts_enabled:
            raise SynthesisError("F5-TTS is disabled (set F5_TTS_ENABLED=true)")
        if request.voice_reference is None:
            raise SynthesisError("F5-TTS requires a reference wav")

        model = self._model()
        out = Path(request.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            model.infer(
                ref_file=str(request.voice_reference),
                ref_text="",
                gen_text=request.text,
                file_wave=str(out),
                remove_silence=True,
            )
        except Exception as exc:
            raise SynthesisError(f"F5-TTS synthesis failed: {exc}") from exc

        return SynthesisResult(
            path=out, duration=ffmpeg.duration_of(out), model=self.name,
            voice_cloned=True, sample_rate=24000,
        )
