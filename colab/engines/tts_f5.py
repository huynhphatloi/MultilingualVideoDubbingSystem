"""F5-TTS - flow-matching cloning, and the cleanest A/B in the whole set.

Two checkpoints of the *same architecture* are registered:

* ``f5_base``  - the official English/Chinese release, trained on Emilia.
* ``f5_vi``    - a Vietnamese fine-tune trained on ~1000h of ViVoice.

Registering both is the point. viXTTS vs XTTS-v2 already shows what a
Vietnamese fine-tune does to one architecture; f5_vi vs f5_base shows it for a
second, different one. If both pairs move the same way, the finding is about
fine-tuning; if they diverge, it is about the architecture. One engine alone
cannot tell those apart, which is why the "extra" base model earns its slot.

Architecturally F5 differs from XTTS in a way that shows up in the numbers:
XTTS decodes autoregressively (cost grows with output length, and it can
babble on short inputs), while F5 is a flow-matching DiT with a fixed number of
solver steps (cost is flat per call). Expect F5 to lose on very short lines in
RTF and win on long ones - and subtitle lines are mostly short.
"""
from __future__ import annotations

import time
from pathlib import Path

from .tts_base import Spoken, TTSEngine


class _F5Base(TTSEngine):
    voice_cloning = True
    vram_gb = 3.0
    #: HF repo holding the .safetensors/.pt checkpoint.
    repo: str = ""
    #: File inside that repo.
    ckpt_file: str = ""
    #: Vocabulary file, or "" for the model's default.
    vocab_file: str = ""

    def __init__(self) -> None:
        super().__init__()
        self._api = None
        self._sr = 24000
        # F5 conditions on reference audio AND its transcript. Transcribing the
        # same 6-second reference once per line would dominate the run, so the
        # transcript is resolved once per reference file.
        self._ref_text: dict[str, str] = {}

    def load(self) -> None:
        import torch
        from f5_tts.api import F5TTS
        from huggingface_hub import hf_hub_download

        ckpt = hf_hub_download(self.repo, self.ckpt_file) if self.repo else ""
        vocab = hf_hub_download(self.repo, self.vocab_file) if self.vocab_file else ""
        self._api = F5TTS(
            model="F5TTS_v1_Base",
            ckpt_file=ckpt,
            vocab_file=vocab,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        self._sr = getattr(self._api, "target_sample_rate", 24000)

    def unload(self) -> None:
        import gc

        import torch

        self._api = None
        self._ref_text.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        super().unload()

    def speak(self, text: str, language: str, out_path: Path,
              reference_wav: Path | None = None, speed: float = 1.0) -> Spoken:
        import numpy as np
        import soundfile as sf

        self.ensure_loaded()
        if reference_wav is None:
            raise RuntimeError(f"{self.name} clones a voice and needs a reference wav")

        # "" tells F5 to transcribe the reference itself (Whisper under the
        # hood). Caching the result keeps that cost at once per speaker.
        key = str(reference_wav)
        ref_text = self._ref_text.get(key, "")

        started = time.perf_counter()
        with self._lock:
            wav, sr, _ = self._api.infer(
                ref_file=key,
                ref_text=ref_text,
                gen_text=text,
                speed=float(speed) if speed else 1.0,
                remove_silence=True,
                file_wave=str(out_path),
            )
            if not ref_text:
                # infer() stores whatever it transcribed; reuse it next line.
                self._ref_text[key] = getattr(self._api, "ref_text", "") or ""
        compute = time.perf_counter() - started

        wav = np.asarray(wav, dtype="float32")
        if not out_path.exists():
            sf.write(str(out_path), wav, sr)
        return Spoken(
            wav_path=out_path, sample_rate=sr, seconds=len(wav) / sr,
            compute_seconds=compute, voice_cloned=True, engine=self.name,
            notes={"ref_text_cached": bool(ref_text)},
        )


class F5Vietnamese(_F5Base):
    name = "f5_vi"
    title = "F5-TTS Vietnamese (ViVoice ~1000h fine-tune)"
    languages = ("vi",)
    repo = "hynt/F5-TTS-Vietnamese-ViVoice"
    ckpt_file = "model_last.pt"
    vocab_file = "vocab.txt"


class F5Base(_F5Base):
    name = "f5_base"
    title = "F5-TTS v1 Base (official, en/zh)"
    languages = ("en", "zh")
    repo = "SWivid/F5-TTS"
    ckpt_file = "F5TTS_v1_Base/model_1250000.safetensors"
