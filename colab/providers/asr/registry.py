"""Speech recognisers this build knows about.

Language lists are copied from each model's own card or source, never inferred:

* Whisper - the 99 languages of the multilingual checkpoints cover every code in
  the application table; the `.en` and `distil-` checkpoints are English-only.
* SeamlessM4T - the source-speech column of the M4T language table. Malay is
  text-only there and Sinhala is absent, so both are excluded.
* MMS - the adapter files published in facebook/mms-1b-all. Eight application
  languages have no adapter, and five need a script-qualified adapter name.
* SenseVoiceSmall - "Mandarin, Cantonese, English, Japanese, and Korean";
  Cantonese has no code in the application table, so four remain.
* Parakeet - v3 lists 25 European languages, v2 is English-only.
"""
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
SENSEVOICE = REGISTRY.add_provider(ProviderSpec(
    id="sensevoice",
    task="asr",
    display_name="SenseVoice",
    homepage="https://github.com/FunAudioLLM/SenseVoice",
    notes="Fast non-autoregressive recogniser for East Asian languages.",
))
PARAKEET = REGISTRY.add_provider(ProviderSpec(
    id="parakeet",
    task="asr",
    display_name="NVIDIA Parakeet",
    homepage="https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3",
    notes="Needs the NeMo toolkit, which is a large install.",
))

#: checkpoint name -> (Hugging Face repo, English-only, label)
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
    #: Source-speech column of the M4T table, intersected with this project's
    #: languages: Malay is target-text only and Sinhala is not listed.
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
    #: Every application language whose adapter exists in the checkpoint.
    #: Only Sinhala has none. Seven others need a script-qualified or
    #: differently-coded adapter name, which providers.asr.mms maps.
    languages=L.exclude("si"),
    multilingual=True,
    license="cc-by-nc-4.0",
    notes="Non-commercial licence. One CTC adapter per language, so the source "
          "language must be named. Windows come from voice-activity detection.",
    supports_language_detection=False,
    supports_timestamps=False,
    module="providers.asr.mms",
))

REGISTRY.add(ModelSpec(
    id="sensevoice_small",
    task="asr",
    provider="sensevoice",
    display_name="SenseVoiceSmall",
    repo_id="FunAudioLLM/SenseVoiceSmall",
    #: The card lists Mandarin, Cantonese, English, Japanese and Korean.
    #: Cantonese has no code in this application's table.
    languages=L.subset("zh", "en", "ja", "ko"),
    multilingual=True,
    license="FunASR Model Open Source License (see the model card)",
    optional_package="funasr",
    import_name="funasr",
    experimental=True,
    notes="Weights are under the FunASR model licence rather than a standard "
          "open-source one. Emits rich transcription tags, which this provider "
          "strips.",
    supports_language_detection=True,
    supports_timestamps=False,
    module="providers.asr.sensevoice",
))

REGISTRY.add(ModelSpec(
    id="parakeet_tdt_0.6b_v3",
    task="asr",
    provider="parakeet",
    display_name="Parakeet TDT 0.6B v3 (European)",
    repo_id="nvidia/parakeet-tdt-0.6b-v3",
    #: The 25 languages the card lists, intersected with this table. Estonian,
    #: Latvian, Lithuanian and Maltese are supported by the model but have no
    #: code here.
    languages=L.subset(
        "bg", "hr", "cs", "da", "nl", "en", "fi", "fr", "de", "el", "hu", "it",
        "pl", "pt", "ro", "sk", "sl", "es", "sv", "ru", "uk",
    ),
    multilingual=True,
    license="cc-by-4.0",
    optional_package="nemo_toolkit[asr]",
    import_name="nemo",
    experimental=True,
    notes="Transcribes without being told the language and returns segment "
          "timestamps, but never reports which language it heard - so the "
          "source language still has to be named for the translation stage. "
          "NeMo is a heavy install that pins its own transformers version.",
    supports_language_detection=False,
    supports_timestamps=True,
    module="providers.asr.parakeet",
))

REGISTRY.add(ModelSpec(
    id="parakeet_tdt_0.6b_v2",
    task="asr",
    provider="parakeet",
    display_name="Parakeet TDT 0.6B v2 (English)",
    repo_id="nvidia/parakeet-tdt-0.6b-v2",
    languages=L.subset("en"),
    multilingual=False,
    license="cc-by-4.0",
    optional_package="nemo_toolkit[asr]",
    import_name="nemo",
    experimental=True,
    notes="English only, by the model card.",
    supports_language_detection=False,
    supports_timestamps=True,
    module="providers.asr.parakeet",
))

#: The checkpoint used when a request omits the model. Kept as the old default.
DEFAULT_MODEL = "small"
