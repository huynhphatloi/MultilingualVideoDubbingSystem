"""pyannote speaker diarization."""
from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Dict, List, Optional

from core.errors import MissingCredential, ProviderFailure
from core.runtime import device
from dubflow_core.segments import merge_turns
from providers.base import ModelSpec
from providers.diarization.base import DiarizationProvider

TOKEN_ENV = ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN")


def _token() -> str:
    for name in TOKEN_ENV:
        value = os.getenv(name, "").strip()
        if value:
            return value
    raise MissingCredential(
        "pyannote's weights are gated. Accept the conditions at "
        "huggingface.co/pyannote/speaker-diarization-3.1 and "
        "huggingface.co/pyannote/segmentation-3.0, then set HF_TOKEN in the "
        "notebook before starting the API cell."
    )


class PyannoteProvider(DiarizationProvider):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__(spec)
        from pyannote.audio import Pipeline

        parameters = inspect.signature(Pipeline.from_pretrained).parameters
        auth = (
            {"use_auth_token": _token()}
            if "use_auth_token" in parameters and "token" not in parameters
            else {"token": _token()}
        )
        pipeline = Pipeline.from_pretrained(
            spec.repo_id or "pyannote/speaker-diarization-3.1",
            **auth,
        )
        if pipeline is None:
            raise ProviderFailure(
                "pyannote returned no pipeline. The usual cause is a token that "
                "has not accepted the model's conditions."
            )
        if device() == "cuda":
            import torch

            pipeline.to(torch.device("cuda"))
        self.pipeline = pipeline

    def diarize(
        self,
        audio: Path,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> List[Dict]:
        arguments = {}
        if min_speakers:
            arguments["min_speakers"] = int(min_speakers)
        if max_speakers:
            arguments["max_speakers"] = int(max_speakers)
        result = self.pipeline(str(audio), **arguments)
        annotation = getattr(result, "exclusive_speaker_diarization", None)
        if annotation is None:
            annotation = getattr(result, "speaker_diarization", result)
        turns = [
            {"speaker_id": str(speaker), "start": float(turn.start), "end": float(turn.end)}
            for turn, _, speaker in annotation.itertracks(yield_label=True)
        ]
        if not turns:
            raise ProviderFailure("Diarization found no speech")
        return merge_turns(turns)


def cache_key(spec: ModelSpec, **_: object) -> str:
    return f"diarization:{spec.id}"


def build(spec: ModelSpec, **_: object) -> DiarizationProvider:
    return PyannoteProvider(spec)
