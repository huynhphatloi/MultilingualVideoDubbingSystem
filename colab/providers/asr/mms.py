"""Meta MMS speech recognition.

MMS is one CTC encoder plus a per-language adapter, so the language has to be
named - it cannot be detected. The adapter names are the ones actually published
in facebook/mms-1b-all: most are plain ISO-639-3, seven are not, and Sinhala has
no adapter at all.

Licence note: CC-BY-NC-4.0. The registry records that, and the capabilities
endpoint reports it, so a commercial deployment can filter this out.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from core.errors import ProviderFailure, UnsupportedLanguage
from core.runtime import device
from dubflow_core import languages as L
from providers.asr.base import WindowedASR
from providers.base import ModelSpec

#: Application code -> adapter name in facebook/mms-1b-all, where the adapter is
#: not simply the ISO-639-3 code from the language table. Verified against the
#: adapter files published in the checkpoint.
ADAPTER_OVERRIDES: Dict[str, str] = {
    "zh": "cmn-script_simplified",
    "ar": "ara",
    "fa": "fas",
    "ur": "urd-script_arabic",
    "sr": "srp-script_cyrillic",
    "ms": "zlm",
    "mn": "mon",
}
#: No adapter exists for these application languages.
UNSUPPORTED = ("si",)


def adapter_for(language: str) -> str:
    if language in UNSUPPORTED or not L.is_supported(language):
        raise UnsupportedLanguage(
            f"MMS has no speech-recognition adapter for '{language}'."
        )
    return ADAPTER_OVERRIDES.get(language, L.iso3(language))


class MMSASRProvider(WindowedASR):
    max_window_seconds = 20.0

    def __init__(self, spec: ModelSpec, language: str) -> None:
        super().__init__(spec)
        from transformers import AutoProcessor, Wav2Vec2ForCTC

        self.language = language
        self.adapter = adapter_for(language)
        repo = spec.repo_id or "facebook/mms-1b-all"
        self.processor = AutoProcessor.from_pretrained(repo, target_lang=self.adapter)
        self.model = Wav2Vec2ForCTC.from_pretrained(
            repo, target_lang=self.adapter, ignore_mismatched_sizes=True
        )
        # Loading the adapter twice is harmless and covers both transformers
        # paths: the one that applies target_lang at from_pretrained and the one
        # that expects an explicit load_adapter call.
        self.model.load_adapter(self.adapter)
        self.model = self.model.to(device()).eval()

    @property
    def name(self) -> str:
        return f"{self.spec.repo_id}#{self.adapter}"

    def transcribe_window(self, clip: Path, language: str) -> str:
        import soundfile
        import torch

        if language != self.language:  # pragma: no cover - the slot key prevents it
            raise ProviderFailure(
                f"This MMS instance holds the '{self.language}' adapter, not '{language}'"
            )
        waveform, rate = soundfile.read(str(clip), dtype="float32")
        inputs = self.processor(waveform, sampling_rate=rate, return_tensors="pt")
        inputs = {name: value.to(device()) for name, value in inputs.items()}
        with torch.inference_mode():
            logits = self.model(**inputs).logits
        predicted = logits.argmax(dim=-1)[0]
        return self.processor.decode(predicted)


def cache_key(spec: ModelSpec, language: Optional[str] = None) -> str:
    #: The adapter is part of what is loaded, so it is part of the slot key.
    return f"asr:{spec.id}:{language or 'none'}"


def build(spec: ModelSpec, language: Optional[str] = None) -> WindowedASR:
    if not language:
        raise UnsupportedLanguage(
            "MMS cannot detect the spoken language: it loads one language adapter "
            "at a time. Set source_language explicitly."
        )
    return MMSASRProvider(spec, language)
