"""Speech recognition model registry."""
from __future__ import annotations

from dubflow_core import languages as L
from providers.base import ModelSpec, ProviderSpec, Registry

REGISTRY = Registry("asr")

WHISPER = REGISTRY.add_provider(ProviderSpec(
    id="faster_whisper",
    task="asr",
    display_name="Whisper (faster-whisper)",
    homepage="https://github.com/SYSTRAN/faster-whisper",
    notes="Timestamps and language detection; part of the base install.",
))
SEAMLESS = REGISTRY.add_provider(ProviderSpec(
    id="seamless",
    task="asr",
    display_name="SeamlessM4T",
    homepage="https://huggingface.co/facebook/seamless-m4t-v2-large",
    notes="Speech-to-text mode of the translation model. No timestamps: the "
          "pipeline windows the audio first.",
))
MMS = REGISTRY.add_provider(ProviderSpec(
    id="mms",
    task="asr",
    display_name="Meta MMS",
    homepage="https://huggingface.co/facebook/mms-1b-all",
    notes="1162 language adapters. CC-BY-NC-4.0: research use only.",
))

_WHISPER_CHECKPOINTS = (
    ("tiny", "Systran/faster-whisper-tiny", False, "Tiny"),
    ("tiny.en", "Systran/faster-whisper-tiny.en", True, "Tiny (English)"),
    ("base", "Systran/faster-whisper-base", False, "Base"),
    ("base.en", "Systran/faster-whisper-base.en", True, "Base (English)"),
    ("small", "Systran/faster-whisper-small", False, "Small"),
    ("small.en", "Systran/faster-whisper-small.en", True, "Small (English)"),
    ("medium", "Systran/faster-whisper-medium", False, "Medium"),
    ("medium.en", "Systran/faster-whisper-medium.en", True, "Medium (English)"),
    ("large-v2", "Systran/faster-whisper-large-v2", False, "Large v2"),
    ("large-v3", "Systran/faster-whisper-large-v3", False, "Large v3"),
    ("large-v3-turbo", "mobiuslabsgmbh/faster-whisper-large-v3-turbo", False,
     "Large v3 Turbo"),
    ("distil-small.en", "Systran/faster-distil-whisper-small.en", True,
     "Distil Small (English)"),
    ("distil-medium.en", "Systran/faster-distil-whisper-medium.en", True,
     "Distil Medium (English)"),
    ("distil-large-v3", "Systran/faster-distil-whisper-large-v3", True,
     "Distil Large v3 (English)"),
)

REGISTRY.extend(
    ModelSpec(
        id=checkpoint,
        task="asr",
        provider="faster_whisper",
        display_name=f"Whisper {label}",
        repo_id=repo,
        languages=L.subset("en") if english_only else L.ALL,
        multilingual=not english_only,
        license="mit",
        supports_language_detection=not english_only,
        supports_timestamps=True,
        module="providers.asr.faster_whisper",
    )
    for checkpoint, repo, english_only, label in _WHISPER_CHECKPOINTS
)

REGISTRY.add(ModelSpec(
    id="seamless_asr",
    task="asr",
    provider="seamless",
    display_name="SeamlessM4T (speech to text)",
    repo_id="facebook/hf-seamless-m4t-medium",
    languages=L.exclude("ms", "si"),
    multilingual=True,
    license="cc-by-nc-4.0",
    notes="Non-commercial licence. Transcribes windows found by voice-activity "
          "detection, so its segment boundaries are coarser than Whisper's.",
    supports_language_detection=False,
    supports_timestamps=False,
    module="providers.asr.seamless",
))

REGISTRY.add(ModelSpec(
    id="mms_asr",
    task="asr",
    provider="mms",
    display_name="MMS-1B-all",
    repo_id="facebook/mms-1b-all",
    languages=L.exclude("si"),
    multilingual=True,
    license="cc-by-nc-4.0",
    notes="Non-commercial licence. One CTC adapter per language, so the source "
          "language must be named. Windows come from voice-activity detection.",
    supports_language_detection=False,
    supports_timestamps=False,
    module="providers.asr.mms",
))

DEFAULT_MODEL = "small"
