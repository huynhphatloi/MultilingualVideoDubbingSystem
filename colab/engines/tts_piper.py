"""Piper - the speed ceiling, and the argument for keeping a CPU option.

Piper is a small VITS variant compiled to ONNX. It runs faster than real time
on a laptop CPU with no GPU at all, which makes it the only engine here that
could serve the whole pipeline for free and offline. It does not clone, and its
Vietnamese voices are trained on modest corpora, so nobody expects it to win on
naturalness.

Its job in the comparison is to price the alternative: if a 20 MB CPU model
lands within a small margin of a 3 GB GPU model on intelligibility, the GPU
spend needs a better justification than "it is bigger".

Voices are named ``<lang>_<REGION>-<dataset>-<quality>`` on the Hub, e.g.
``vi_VN-vais1000-medium``. One ONNX + one JSON per voice; both are downloaded
on first use and cached.
"""
from __future__ import annotations

import time
from pathlib import Path

from .tts_base import Spoken, TTSEngine

#: One voice per language, picked for the best available quality tier.
_VOICES = {
    "vi": "vi_VN-vais1000-medium",
    "en": "en_US-lessac-medium",
    "fr": "fr_FR-siwis-medium",
    "de": "de_DE-thorsten-medium",
    "es": "es_ES-davefx-medium",
    "it": "it_IT-riccardo-x_low",
    "pt": "pt_BR-faber-medium",
    "ru": "ru_RU-irina-medium",
    "nl": "nl_NL-mls-medium",
    "pl": "pl_PL-darkman-medium",
    "tr": "tr_TR-fettah-medium",
    "zh": "zh_CN-huayan-medium",
}
_REPO = "rhasspy/piper-voices"


class PiperTTS(TTSEngine):
    name = "piper"
    title = "Piper (ONNX VITS, CPU)"
    voice_cloning = False
    languages = tuple(_VOICES)
    vram_gb = 0.0  # runs on CPU on purpose - that is the finding

    def __init__(self) -> None:
        super().__init__()
        self._voices: dict[str, object] = {}
        self._sr = 22050

    def load(self) -> None:
        pass  # per-voice, like MMS

    def unload(self) -> None:
        self._voices.clear()
        super().unload()

    def _voice(self, language: str):  # noqa: ANN202
        from huggingface_hub import hf_hub_download
        from piper import PiperVoice

        code = (language or "").lower().split("-")[0]
        if code not in _VOICES:
            raise RuntimeError(f"piper has no voice configured for '{language}'")
        if code not in self._voices:
            voice = _VOICES[code]
            lang_dir = voice.split("-")[0]           # vi_VN
            family = lang_dir.split("_")[0]          # vi
            quality = voice.rsplit("-", 1)[1]        # medium
            base = f"{family}/{lang_dir}/{voice.split('-')[1]}/{quality}/{voice}"
            onnx = hf_hub_download(_REPO, f"{base}.onnx")
            hf_hub_download(_REPO, f"{base}.onnx.json")
            self._voices[code] = PiperVoice.load(onnx)
        return self._voices[code]

    def speak(self, text: str, language: str, out_path: Path,
              reference_wav: Path | None = None, speed: float = 1.0) -> Spoken:
        import wave

        self.ensure_loaded()
        voice = self._voice(language)

        started = time.perf_counter()
        with self._lock, wave.open(str(out_path), "wb") as fh:
            # Piper's length_scale is inverse to speed, same convention as VITS.
            voice.synthesize_wav(
                text, fh,
                syn_config=_syn_config(voice, 1.0 / float(speed or 1.0)),
            )
        compute = time.perf_counter() - started

        with wave.open(str(out_path), "rb") as fh:
            sr = fh.getframerate()
            frames = fh.getnframes()
        self._sr = sr
        return Spoken(
            wav_path=out_path, sample_rate=sr, seconds=frames / sr,
            compute_seconds=compute, voice_cloned=False, engine=self.name,
            notes={"voice": _VOICES[(language or "").lower().split("-")[0]]},
        )


def _syn_config(voice, length_scale: float):  # noqa: ANN001, ANN202
    """Build a SynthesisConfig, tolerating the 1.2 -> 1.3 API rename."""
    try:
        from piper import SynthesisConfig
    except ImportError:
        return None
    cfg = SynthesisConfig(length_scale=length_scale)
    return cfg
