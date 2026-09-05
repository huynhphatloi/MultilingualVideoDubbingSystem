"""NVIDIA Parakeet through the NeMo toolkit.

Parakeet returns its own segment timestamps, so it is a timestamped provider
rather than a windowed one. v3 identifies the language itself; v2 is English
only, which the registry records.
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


class ParakeetProvider(ASRProvider):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__(spec)
        import nemo.collections.asr as nemo_asr

        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=spec.repo_id)
        if device() == "cuda":
            self.model = self.model.cuda()
        self.model.eval()

    def transcribe(
        self,
        audio: Path,
        language: Optional[str],
        windows: Optional[Sequence[Window]] = None,
    ) -> Transcript:
        if language is None:
            language = self.detect_language(audio)
        if not L.is_supported(language):  # pragma: no cover - validated earlier
            raise ProviderFailure(f"Unsupported language '{language}'")
        # Parakeet picks the language itself and offers no way to force one;
        # `language` is carried only so the translation stage knows what it is
        # translating from, which is why the registry says it cannot detect.
        output = self.model.transcribe([str(audio)], timestamps=True)
        if not output:
            raise ProviderFailure("Parakeet returned no transcription")
        stamps = getattr(output[0], "timestamp", None) or {}
        rows = stamps.get("segment") or []
        segments = []
        for row in rows:
            text = str(row.get("segment") or "").strip()
            if text:
                segments.append(
                    make_segment(len(segments), row["start"], row["end"], text)
                )
        if not segments:
            text = str(getattr(output[0], "text", "")).strip()
            if not text:
                raise ProviderFailure("Parakeet found no speech in the audio")
            raise ProviderFailure(
                "Parakeet returned text without segment timestamps, which this "
                "pipeline needs to place the dub. Use a Whisper checkpoint instead."
            )
        return Transcript(language, segments)


def cache_key(spec: ModelSpec, language: Optional[str] = None) -> str:
    return f"asr:{spec.id}"


def build(spec: ModelSpec, language: Optional[str] = None) -> ASRProvider:
    return ParakeetProvider(spec)
