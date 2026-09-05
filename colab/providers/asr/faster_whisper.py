"""faster-whisper: the baseline recogniser, unchanged in behaviour.

Its `id` is the checkpoint name faster-whisper resolves ("large-v3-turbo"), and
`repo_id` records which Hugging Face repository that resolves to.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from core.errors import ProviderFailure
from core.runtime import device
from dubflow_core import languages as L
from dubflow_core.segments import make_segment
from providers.asr.base import ASRProvider, Transcript, Window
from providers.base import ModelSpec


class FasterWhisperProvider(ASRProvider):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__(spec)
        from faster_whisper import WhisperModel

        chosen = device()
        self.model = WhisperModel(
            spec.id,
            device=chosen,
            compute_type="float16" if chosen == "cuda" else "int8",
        )

    def detect_language(self, audio: Path) -> str:
        _, info = self.model.transcribe(str(audio), vad_filter=True, beam_size=1)
        return str(info.language)

    def transcribe(
        self,
        audio: Path,
        language: Optional[str],
        windows: Optional[Sequence[Window]] = None,
    ) -> Transcript:
        raw, info = self.model.transcribe(
            str(audio),
            language=language,
            vad_filter=True,
            beam_size=5,
        )
        segments = []
        for piece in raw:
            text = piece.text.strip()
            if text:
                segments.append(make_segment(len(segments), piece.start, piece.end, text))
        detected = language or str(info.language)
        if not L.is_supported(detected):
            raise ProviderFailure(
                f"Whisper detected '{detected}', which this application does not "
                f"support. Set source_language to one of the supported codes."
            )
        return Transcript(detected, segments)


def cache_key(spec: ModelSpec, language: Optional[str] = None) -> str:
    return f"asr:{spec.id}"


def build(spec: ModelSpec, language: Optional[str] = None) -> ASRProvider:
    return FasterWhisperProvider(spec)
