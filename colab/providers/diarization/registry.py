"""Diarization providers."""
from __future__ import annotations

from providers.base import ModelSpec, ProviderSpec, Registry

REGISTRY = Registry("diarization")

REGISTRY.add_provider(ProviderSpec(
    id="pyannote",
    task="diarization",
    display_name="pyannote.audio",
    homepage="https://huggingface.co/pyannote/speaker-diarization-3.1",
    notes="Gated weights: accept the conditions on the model page and provide a "
          "Hugging Face token.",
))

REGISTRY.add(ModelSpec(
    id="pyannote_3_1",
    task="diarization",
    provider="pyannote",
    display_name="pyannote speaker-diarization 3.1",
    repo_id="pyannote/speaker-diarization-3.1",
    languages=(),
    multilingual=True,
    license="mit",
    optional_package="pyannote.audio",
    import_name="pyannote.audio",
    required_env="HF_TOKEN",
    notes="Wants mono 16 kHz audio, which the extract stage already produces. "
          "The weights are MIT but gated behind a click-through on the Hub.",
    module="providers.diarization.pyannote",
))

DEFAULT_MODEL = "pyannote_3_1"
