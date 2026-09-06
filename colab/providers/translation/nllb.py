"""NLLB-200. Language codes are FLORES-200, from the shared language table."""
from __future__ import annotations

import os
from typing import List, Sequence

from core.errors import ProviderFailure
from core.runtime import device
from dubflow_core import languages as L
from providers.base import ModelSpec
from providers.translation.base import TranslationProvider

MODEL_ENV = "TRANSLATION_MODEL"


class NLLBProvider(TranslationProvider):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__(spec)
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        override = os.getenv(MODEL_ENV, "").strip()
        self.repo = override if (override and spec.id == "nllb") else spec.repo_id
        self.tokenizer = AutoTokenizer.from_pretrained(self.repo)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.repo).to(device()).eval()

    @property
    def name(self) -> str:
        return self.repo

    def translate(self, texts: Sequence[str], source: str, target: str) -> List[str]:
        import torch

        self.tokenizer.src_lang = L.flores(source)
        encoded = self.tokenizer(
            list(texts), return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        encoded = {name: value.to(device()) for name, value in encoded.items()}
        forced = self.tokenizer.convert_tokens_to_ids(L.flores(target))
        with torch.inference_mode():
            output = self.model.generate(
                **encoded, forced_bos_token_id=forced, max_new_tokens=256, num_beams=4
            )
        decoded = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        if len(decoded) != len(texts):
            raise ProviderFailure("NLLB returned the wrong number of translations")
        return [text.strip() for text in decoded]


def cache_key(spec: ModelSpec, **_: object) -> str:
    return f"translation:{spec.id}"


def build(spec: ModelSpec, **_: object) -> TranslationProvider:
    return NLLBProvider(spec)
