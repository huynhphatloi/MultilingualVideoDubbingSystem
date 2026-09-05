"""CosyVoice 2, installed from its own repository.

It is not on PyPI: the notebook clones the repository, and COSYVOICE_ROOT points
at that checkout so its bundled third_party/Matcha-TTS can go on sys.path.
Zero-shot cloning wants the transcript of the reference clip as well as the
audio, which the pipeline has from the recognition stage.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from core.errors import MissingDependency, ProviderFailure
from core.media import write_waveform
from providers.base import ModelSpec
from providers.tts.base import SpeechRequest, TTSProvider

ROOT_ENV = "COSYVOICE_ROOT"
#: The zero-shot prompt is resampled to 16 kHz by CosyVoice's own loader.
PROMPT_SAMPLE_RATE = 16000


def _prepare_path() -> Path:
    root = Path(os.getenv(ROOT_ENV, "/content/CosyVoice"))
    if not (root / "cosyvoice").is_dir():
        raise MissingDependency(
            f"CosyVoice (git clone --recursive https://github.com/FunAudioLLM/CosyVoice "
            f"and set {ROOT_ENV})",
            "CosyVoice 2",
        )
    for entry in (root, root / "third_party" / "Matcha-TTS"):
        if str(entry) not in sys.path:
            sys.path.insert(0, str(entry))
    return root


class CosyVoiceProvider(TTSProvider):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__(spec)
        _prepare_path()
        from cosyvoice.cli.cosyvoice import CosyVoice2
        from huggingface_hub import snapshot_download

        folder = os.getenv("COSYVOICE_MODEL") or snapshot_download(
            spec.repo_id or "FunAudioLLM/CosyVoice2-0.5B"
        )
        self.model = CosyVoice2(folder, load_jit=False, load_trt=False, fp16=False)

    @property
    def name(self) -> str:
        return self.spec.repo_id or "FunAudioLLM/CosyVoice2-0.5B"

    def synthesize(self, request: SpeechRequest, out: Path) -> None:
        import torch
        from cosyvoice.utils.file_utils import load_wav

        reference = self.require_reference(request)
        reference_text = self.require_reference_text(request)
        prompt = load_wav(str(reference), PROMPT_SAMPLE_RATE)
        pieces = [
            result["tts_speech"]
            for result in self.model.inference_zero_shot(
                request.text, reference_text, prompt, stream=False
            )
        ]
        if not pieces:
            raise ProviderFailure("CosyVoice produced no audio")
        audio = torch.cat(pieces, dim=1).squeeze(0).cpu().numpy()
        write_waveform(audio, int(self.model.sample_rate), out, request.speed)


def cache_key(spec: ModelSpec, language: Optional[str] = None) -> str:
    return f"tts:{spec.id}"


def build(spec: ModelSpec, language: Optional[str] = None, **_: object) -> TTSProvider:
    return CosyVoiceProvider(spec)
