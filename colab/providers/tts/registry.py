"""Speech generation model registry."""
from __future__ import annotations

from dubflow_core import languages as L
from providers.base import ModelSpec, ProviderSpec, Registry

REGISTRY = Registry("tts")

for provider in (
    ProviderSpec("mms", "tts", "Meta MMS-TTS",
                 "https://huggingface.co/facebook/mms-tts",
                 "One small VITS checkpoint per language. Part of the base install."),
    ProviderSpec("edge", "tts", "Microsoft Edge TTS",
                 "https://github.com/rany2/edge-tts",
                 "Cloud voices; needs internet on every call, no GPU.")
):
    REGISTRY.add_provider(provider)


MMS_TTS_OVERRIDES = {"ar": "ara", "fa": "fas", "ms": "zlm"}
MMS_TTS_LANGUAGES = L.subset(
    "en", "vi", "ko", "fr", "de", "es", "pt", "ru", "nl", "pl", "tr", "ar", "hi",
    "id", "th", "hu", "uk", "ro", "sv", "fi", "ms", "fa", "bn", "ta", "te", "sw",
    "tl", "km", "lo", "my", "bg", "ca", "el", "he",
)

REGISTRY.add(ModelSpec(
    id="mms",
    task="tts",
    provider="mms",
    display_name="MMS-TTS",
    repo_id="facebook/mms-tts-<language>",
    languages=MMS_TTS_LANGUAGES,
    multilingual=True,
    license="cc-by-nc-4.0",
    notes="Non-commercial licence. A single stock voice per language; the "
          "16 application languages without a checkpoint are rejected up front.",
    speed_control="native",
    recommended_sample_rate=16000,
    module="providers.tts.mms",
))

REGISTRY.add(ModelSpec(
    id="edge",
    task="tts",
    provider="edge",
    display_name="Edge TTS",
    repo_id=None,
    languages=L.exclude("hy"),
    multilingual=True,
    license="Microsoft service terms (not a model licence)",
    optional_package="edge-tts",
    import_name="edge_tts",
    notes="Sends the text to Microsoft's public endpoint, so it needs internet "
          "and is unsuitable for private material. The most natural non-cloning "
          "option and the only one that needs no GPU.",
    supports_multispeaker=True,
    speed_control="native",
    recommended_sample_rate=24000,
    module="providers.tts.edge",
))

DEFAULT_MODEL = "edge"
ALIASES: dict = {}
