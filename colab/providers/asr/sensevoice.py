"""SenseVoiceSmall through FunASR.

Output is "rich transcription": the text is prefixed with tags such as
`<|en|><|NEUTRAL|><|Speech|><|woitn|>`. The leading tag is the language the
model identified, which is where language detection comes from; the rest is
stripped so the pipeline sees plain text.

Licence note: the weights are under the FunASR Model Open Source License, not a
standard open-source licence. The registry marks the model experimental.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional

from core.errors import ProviderFailure
from core.runtime import device
from providers.asr.base import WindowedASR
from providers.base import ModelSpec

_TAG = re.compile(r"<\|([^|]*)\|>")

#: The language tags SenseVoice emits, mapped onto application codes. Cantonese
#: ("yue") is recognised by the model but has no code in this application.
_TAG_LANGUAGES: Dict[str, str] = {"zh": "zh", "en": "en", "ja": "ja", "ko": "ko"}


class SenseVoiceProvider(WindowedASR):
    #: The card recommends VAD segments of at most 30 s.
    max_window_seconds = 30.0

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__(spec)
        from funasr import AutoModel

        self.model = AutoModel(
            model=spec.repo_id or "FunAudioLLM/SenseVoiceSmall",
            trust_remote_code=False,
            device="cuda:0" if device() == "cuda" else "cpu",
            disable_update=True,
        )

    def _generate(self, clip: Path, language: str) -> str:
        results = self.model.generate(
            input=str(clip),
            cache={},
            language=language,
            use_itn=True,
            batch_size_s=60,
        )
        if not results:
            raise ProviderFailure("SenseVoice returned no result for a speech window")
        return str(results[0].get("text", ""))

    def detect_language(self, audio: Path) -> str:
        raw = self._generate(audio, "auto")
        for tag in _TAG.findall(raw):
            if tag in _TAG_LANGUAGES:
                return _TAG_LANGUAGES[tag]
        raise ProviderFailure(
            "SenseVoice did not report a language this application supports. "
            "Set source_language explicitly."
        )

    def transcribe_window(self, clip: Path, language: str) -> str:
        return strip_tags(self._generate(clip, language))


def strip_tags(text: str) -> str:
    """Remove the rich-transcription markers, keeping the words."""
    return _TAG.sub(" ", text or "").strip()


def cache_key(spec: ModelSpec, language: Optional[str] = None) -> str:
    return f"asr:{spec.id}"


def build(spec: ModelSpec, language: Optional[str] = None) -> WindowedASR:
    return SenseVoiceProvider(spec)
