"""Demucs source separation through its command line."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Dict

from core.errors import ProviderFailure
from core.media import run
from core.runtime import device
from providers.base import ModelSpec
from providers.separation.base import SourceSeparationProvider


class DemucsProvider(SourceSeparationProvider):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__(spec)
        self.model = spec.repo_id or "htdemucs"

    def separate(self, audio: Path, workdir: Path) -> Dict[str, Path]:
        output = workdir / "separated"
        if output.exists():
            shutil.rmtree(output, ignore_errors=True)
        run([
            sys.executable, "-m", "demucs",
            "--two-stems", "vocals",
            "-n", self.model,
            "-d", device(),
            "-o", str(output),
            str(audio),
        ])
        folder = output / self.model / audio.stem
        speech, background = folder / "vocals.wav", folder / "no_vocals.wav"
        if not speech.exists() or not background.exists():
            raise ProviderFailure(
                f"Demucs wrote no stems under {folder}. Check the Colab log for its output."
            )
        return {"speech": speech, "background": background}


def cache_key(spec: ModelSpec, **_: object) -> str:
    return f"separation:{spec.id}"


def build(spec: ModelSpec, **_: object) -> SourceSeparationProvider:
    return DemucsProvider(spec)
