"""SeamlessM4T text-to-text. Language codes are ISO-639-3."""
from __future__ import annotations

import os
from typing import List, Sequence

from core.errors import ProviderFailure
from core.runtime import device
from dubflow_core import languages as L
from providers.asr.seamless import MODEL_ENV, _load_model
from providers.base import ModelSpec
from providers.translation.base import TranslationProvider


class SeamlessTranslationProvider(TranslationProvider):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__(spec)
        from transformers import AutoProcessor

        self.repo = os.getenv(MODEL_ENV, spec.repo_id)
        self.processor = AutoProcessor.from_pretrained(self.repo)
        self.model = _load_model(self.repo).to(device()).eval()

    @property
    def name(self) -> str:
        return self.repo

    def translate(self, texts: Sequence[str], source: str, target: str) -> List[str]:
        import torch

        encoded = self.processor(
            text=list(texts),
            src_lang=L.iso3(source),
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        encoded = {name: value.to(device()) for name, value in encoded.items()}
        with torch.inference_mode():
            output = self.model.generate(
                **encoded, tgt_lang=L.iso3(target), generate_speech=False, num_beams=4
            )
        # Text-only generation returns a wrapper whose first element is tokens.
        sequences = getattr(output, "sequences", None)
        if sequences is None:
            sequences = output[0] if isinstance(output, (tuple, list)) else output
        decoded = self.processor.batch_decode(sequences, skip_special_tokens=True)
        if len(decoded) != len(texts):
            raise ProviderFailure("SeamlessM4T returned the wrong number of translations")
        return [text.strip() for text in decoded]


def cache_key(spec: ModelSpec, **_: object) -> str:
    return f"translation:{spec.id}"


def build(spec: ModelSpec, **_: object) -> TranslationProvider:
    return SeamlessTranslationProvider(spec)
