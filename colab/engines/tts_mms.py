"""MMS-TTS (facebook/mms-tts-<iso3>) - the floor of the comparison.

This is the engine the local pipeline already falls back to, so it belongs in
the table even though nobody expects it to win. It matters for three reasons:

* it is the only engine here that covers ~1100 languages, which is the whole
  reason the router keeps it as a safety net;
* it is a VITS model - one forward pass, no autoregressive decoding - so its
  RTF is 10-30x better than the cloning engines and sets the "how much are we
  paying for quality" baseline;
* it ships exactly ONE voice per language. Every character in a dub is read by
  the same speaker, which is the concrete defect that motivates cloning at all.

Kept deliberately identical in behaviour to ``ai-service/app/services/tts/
mms.py`` so a Colab number is comparable with a local one.
"""
from __future__ import annotations

import time
from pathlib import Path

from .tts_base import Spoken, TTSEngine

#: ISO-639-1 -> the ISO-639-3 code MMS names its checkpoints by. Only the
#: languages this project actually targets; anything else falls through to the
#: 3-letter code the caller passed.
_ISO3 = {
    "vi": "vie", "en": "eng", "fr": "fra", "de": "deu", "es": "spa",
    "it": "ita", "pt": "por", "ru": "rus", "id": "ind", "th": "tha",
    "hi": "hin", "ar": "ara", "tr": "tur", "nl": "nld", "pl": "pol",
    "ko": "kor", "zh": "cmn",
}


def iso3(language: str) -> str:
    code = (language or "").lower().split("-")[0]
    return _ISO3.get(code, code)


class MmsTTS(TTSEngine):
    name = "mms"
    title = "MMS-TTS (Meta, VITS single-speaker)"
    voice_cloning = False
    # "*" because coverage is per-checkpoint: if facebook/mms-tts-<iso3> exists
    # on the Hub the engine speaks it. Enumerating 1100 codes here would be a
    # lie the moment Meta adds one.
    languages = ("*",)
    vram_gb = 0.5

    def __init__(self) -> None:
        super().__init__()
        self._models: dict[str, tuple] = {}
        self._sr = 16000

    def load(self) -> None:
        # Per-language checkpoints: nothing global to load. The real work
        # happens in _for_language, which caches one model per language.
        pass

    def unload(self) -> None:
        import gc

        import torch

        self._models.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        super().unload()

    def _for_language(self, language: str):  # noqa: ANN202
        import torch
        from transformers import VitsModel, VitsTokenizer

        code = iso3(language)
        if code not in self._models:
            repo = f"facebook/mms-tts-{code}"
            try:
                tokenizer = VitsTokenizer.from_pretrained(repo)
                model = VitsModel.from_pretrained(repo)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"no MMS checkpoint for '{language}' ({repo}): {exc}"
                ) from exc
            model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
            self._models[code] = (tokenizer, model)
            self._sr = model.config.sampling_rate
        return self._models[code]

    def speak(self, text: str, language: str, out_path: Path,
              reference_wav: Path | None = None, speed: float = 1.0) -> Spoken:
        import soundfile as sf
        import torch

        self.ensure_loaded()
        tokenizer, model = self._for_language(language)

        started = time.perf_counter()
        with self._lock, torch.inference_mode():
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            # VITS exposes speaking rate as a length scale: bigger = slower, so
            # a request for 1.2x speed is 1/1.2 of the nominal length.
            model.speaking_rate = float(speed) if speed else 1.0
            wav = model(**inputs).waveform[0].detach().cpu().float().numpy()
        compute = time.perf_counter() - started

        sf.write(str(out_path), wav, self._sr)
        return Spoken(
            wav_path=out_path, sample_rate=self._sr, seconds=len(wav) / self._sr,
            compute_seconds=compute, voice_cloned=False, engine=self.name,
            notes={"checkpoint": f"facebook/mms-tts-{iso3(language)}"},
        )
