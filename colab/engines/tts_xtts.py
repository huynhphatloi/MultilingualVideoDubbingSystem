"""XTTS-v2 family: the multilingual cloning baseline, plus its Vietnamese fork.

Two entries share this file because they are the same architecture with
different weights - which is itself one of the findings the comparison should
surface:

* ``xtts_v2``  - Coqui's official checkpoint. 17 languages, zero-shot cloning,
  and **no Vietnamese**. It is the yardstick for "how good does multilingual
  cloning get" and simultaneously the reason a router exists at all.
* ``vixtts``   - a community fine-tune of the same architecture on Vietnamese
  data. Clones, speaks Vietnamese, but loses the other 16 languages.

Cloning cost note: ``get_conditioning_latents`` dominates a call. One speaker
reuses one reference wav across every line they have, so the latents are cached
per reference path - without that cache the engine is roughly 3x slower on a
real job.
"""
from __future__ import annotations

import time
from pathlib import Path

from .tts_base import Spoken, TTSEngine

# XTTS-v2's own language list. Vietnamese is absent on purpose - that is the
# upstream reality, not an omission here.
_XTTS_LANGS = ("en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl",
               "cs", "ar", "zh", "hu", "ko", "ja", "hi")


class _XttsBase(TTSEngine):
    voice_cloning = True
    vram_gb = 2.5
    #: HF repo or a local directory holding config.json + model.pth + vocab.json
    checkpoint: str = ""
    #: Language string handed to XTTS. viXTTS only ever speaks "vi".
    forced_language: str | None = None

    def __init__(self) -> None:
        super().__init__()
        self._model = None
        self._latents: dict[str, tuple] = {}
        self._sr = 24000

    def load(self) -> None:
        import torch
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

        model_dir = Path(self.checkpoint)
        if not model_dir.exists():
            raise RuntimeError(
                f"{self.name}: checkpoint dir {model_dir} not found - run the "
                "download cell in the notebook first."
            )

        config = XttsConfig()
        config.load_json(str(model_dir / "config.json"))
        model = Xtts.init_from_config(config)
        model.load_checkpoint(config, checkpoint_dir=str(model_dir),
                              vocab_path=str(model_dir / "vocab.json"),
                              use_deepspeed=False)
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        self._model = model
        self._sr = getattr(config.audio, "output_sample_rate", 24000)

    def unload(self) -> None:
        import gc

        import torch

        self._model = None
        self._latents.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        super().unload()

    def _conditioning(self, reference_wav: Path):  # noqa: ANN202
        key = str(reference_wav)
        if key not in self._latents:
            self._latents[key] = self._model.get_conditioning_latents(
                audio_path=key, gpt_cond_len=30, max_ref_length=60,
                sound_norm_refs=True,
            )
        return self._latents[key]

    def speak(self, text: str, language: str, out_path: Path,
              reference_wav: Path | None = None, speed: float = 1.0) -> Spoken:
        import numpy as np
        import soundfile as sf
        import torch

        self.ensure_loaded()
        if reference_wav is None:
            raise RuntimeError(f"{self.name} clones a voice and needs a reference wav")

        lang = self.forced_language or language
        gpt_latent, speaker_embedding = self._conditioning(reference_wav)

        started = time.perf_counter()
        with self._lock, torch.inference_mode():
            result = self._model.inference(
                text=text,
                language=lang,
                gpt_cond_latent=gpt_latent,
                speaker_embedding=speaker_embedding,
                temperature=0.3,
                length_penalty=1.0,
                repetition_penalty=5.0,
                top_k=50,
                top_p=0.85,
                speed=speed,
                enable_text_splitting=True,
            )
        compute = time.perf_counter() - started

        wav = np.asarray(result["wav"], dtype="float32")
        sf.write(str(out_path), wav, self._sr)
        return Spoken(
            wav_path=out_path,
            sample_rate=self._sr,
            seconds=len(wav) / self._sr,
            compute_seconds=compute,
            voice_cloned=True,
            engine=self.name,
            notes={"language_used": lang},
        )


class XttsV2(_XttsBase):
    name = "xtts_v2"
    title = "XTTS-v2 (Coqui, official)"
    languages = _XTTS_LANGS
    checkpoint = "checkpoints/xtts_v2"


class ViXtts(_XttsBase):
    name = "vixtts"
    title = "viXTTS (XTTS-v2 fine-tuned on Vietnamese)"
    languages = ("vi",)
    checkpoint = "checkpoints/vixtts"
    forced_language = "vi"
