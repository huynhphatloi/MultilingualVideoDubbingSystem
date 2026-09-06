"""SeamlessM4T speech recognition."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from core.errors import ProviderFailure
from core.runtime import device
from dubflow_core import languages as L
from providers.asr.base import WindowedASR
from providers.base import ModelSpec

MODEL_ENV = "SEAMLESS_MODEL"


class SeamlessASRProvider(WindowedASR):
    max_window_seconds = 20.0

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__(spec)
        from transformers import AutoProcessor

        self.repo = os.getenv(MODEL_ENV, spec.repo_id or "facebook/hf-seamless-m4t-medium")
        self.processor = AutoProcessor.from_pretrained(self.repo)
        self.model = _load_model(self.repo).to(device()).eval()

    @property
    def name(self) -> str:
        return self.repo

    def transcribe_window(self, clip: Path, language: str) -> str:
        import soundfile
        import torch

        waveform, rate = soundfile.read(str(clip), dtype="float32")
        inputs = self.processor(audios=waveform, sampling_rate=rate, return_tensors="pt")
        inputs = {name: value.to(device()) for name, value in inputs.items()}
        with torch.inference_mode():
            output = self.model.generate(
                **inputs, tgt_lang=L.iso3(language), generate_speech=False
            )
        sequences = getattr(output, "sequences", None)
        if sequences is None:
            sequences = output[0] if isinstance(output, (tuple, list)) else output
        decoded = self.processor.batch_decode(sequences, skip_special_tokens=True)
        if not decoded:
            raise ProviderFailure("SeamlessM4T returned no text for a speech window")
        return decoded[0]


def _load_model(repo: str):  # noqa: ANN202
    """v1 and v2 have different classes; the repo name says which."""
    if "v2" in repo:
        from transformers import SeamlessM4Tv2Model

        return SeamlessM4Tv2Model.from_pretrained(repo)
    from transformers import SeamlessM4TModel

    return SeamlessM4TModel.from_pretrained(repo)


def cache_key(spec: ModelSpec, language: Optional[str] = None) -> str:
    return f"asr:{spec.id}"


def build(spec: ModelSpec, language: Optional[str] = None) -> WindowedASR:
    return SeamlessASRProvider(spec)
