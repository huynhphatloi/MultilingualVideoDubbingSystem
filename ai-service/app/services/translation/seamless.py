"""SeamlessM4T adapter - the fallback engine (and a benchmark alternative).

Used text-to-text only; the pipeline deliberately keeps translation and speech
synthesis as separate steps so each can be swapped independently.
"""
from __future__ import annotations

import logging

from app.core import languages
from app.core.config import settings
from app.core.device import resolve_device_for
from app.core.errors import ModelLoadError, TranslationError
from app.services.models_registry import get_or_load
from app.services.translation.base import (
    TranslationEngine,
    TranslationRequest,
    TranslationResult,
)

log = logging.getLogger(__name__)

_BATCH_SIZE = 8
_LENGTH_PENALTIES = (0.7, 1.0, 1.3)


class SeamlessEngine(TranslationEngine):
    name = "seamless"

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.seamless_model

    def _bundle(self):  # noqa: ANN201
        def loader():  # noqa: ANN202
            try:
                import torch
                from transformers import AutoProcessor, SeamlessM4TModel
            except ImportError as exc:  # pragma: no cover
                raise ModelLoadError("transformers/SeamlessM4T is not available") from exc

            processor = AutoProcessor.from_pretrained(self.model_name)
            model = SeamlessM4TModel.from_pretrained(self.model_name)
            device = resolve_device_for("translation")
            model = model.to(device)
            model.eval()
            return {"processor": processor, "model": model, "torch": torch, "device": device}

        return get_or_load(f"seamless::{self.model_name}", loader)

    def supports(self, source_language: str, target_language: str) -> bool:
        try:
            languages.to_iso3(source_language)
            languages.to_iso3(target_language)
            return True
        except Exception:
            return False

    def translate_batch(self, requests: list[TranslationRequest]) -> list[TranslationResult]:
        if not requests:
            return []
        bundle = self._bundle()
        processor, model, torch = bundle["processor"], bundle["model"], bundle["torch"]
        device = bundle["device"]

        results: list[TranslationResult] = []
        for start in range(0, len(requests), _BATCH_SIZE):
            group = requests[start:start + _BATCH_SIZE]
            src = languages.to_iso3(group[0].source_language)
            tgt = languages.to_iso3(group[0].target_language)

            inputs = processor(
                text=[r.text for r in group], src_lang=src, return_tensors="pt", padding=True
            ).to(device)

            budgeted = any(r.char_budget for r in group)
            penalties = _LENGTH_PENALTIES if budgeted else (1.0,)
            per_request: list[list[str]] = [[] for _ in group]

            for penalty in penalties:
                try:
                    with torch.inference_mode():
                        tokens = model.generate(
                            **inputs, tgt_lang=tgt, generate_speech=False,
                            num_beams=settings.translation_num_beams,
                            length_penalty=penalty,
                            max_new_tokens=settings.translation_max_new_tokens,
                        )
                except Exception as exc:
                    raise TranslationError(f"SeamlessM4T generation failed: {exc}") from exc

                sequences = tokens[0] if isinstance(tokens, tuple | list) else tokens
                decoded = processor.batch_decode(sequences, skip_special_tokens=True)
                for i, text in enumerate(decoded):
                    per_request[i].append(text.strip())

            for request, candidates in zip(group, per_request, strict=True):
                unique = list(dict.fromkeys(c for c in candidates if c))
                if not unique:
                    raise TranslationError("SeamlessM4T returned no output")
                best = (min(unique, key=lambda c: abs(len(c) - request.char_budget))
                        if request.char_budget else unique[0])
                results.append(TranslationResult(
                    text=best, engine=self.name, candidates=unique,
                    char_budget=request.char_budget,
                ))
        return results
