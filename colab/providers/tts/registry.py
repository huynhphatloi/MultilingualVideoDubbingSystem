"""Speech engines this build knows about.

Every language list below was checked against the engine itself, not assumed:

* MMS-TTS - the `facebook/mms-tts-<iso3>` repositories that actually exist. 34
  of the 50 application languages have one; Japanese, Chinese, Italian, Czech,
  Danish and Norwegian are among those that do not.
* Edge - the locale list the Microsoft voice service returns. Every application
  language except Armenian has a voice; Norwegian is `nb` and Tagalog is `fil`,
  which providers.tts.edge maps.
* F5-TTS base - English and Chinese.
* F5 Vietnamese - a single-language Vietnamese checkpoint.
"""
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
                 "Cloud voices; needs internet on every call, no GPU."),
    ProviderSpec("f5", "tts", "F5-TTS",
                 "https://github.com/SWivid/F5-TTS",
                 "Flow-matching voice cloning.")
):
    REGISTRY.add_provider(provider)


#: Application code -> the ISO-639-3 code used by facebook/mms-tts-<code>,
#: where it differs from this project's table.
MMS_TTS_OVERRIDES = {"ar": "ara", "fa": "fas", "ms": "zlm"}
#: The 34 application languages with a published MMS-TTS checkpoint.
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
    #: The service publishes no Armenian voice.
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

REGISTRY.add(ModelSpec(
    id="f5_base",
    task="tts",
    provider="f5",
    display_name="F5-TTS v1 base",
    repo_id="SWivid/F5-TTS",
    languages=L.subset("en", "zh"),
    multilingual=True,
    license="cc-by-nc-4.0",
    optional_package="f5-tts",
    import_name="f5_tts",
    notes="Non-commercial licence. English and Chinese only.",
    supports_voice_cloning=True,
    reference_required=True,
    recommended_sample_rate=24000,
    module="providers.tts.f5",
))

REGISTRY.add(ModelSpec(
    id="f5_vi",
    task="tts",
    provider="f5",
    display_name="F5-TTS Vietnamese",
    repo_id="hynt/F5-TTS-Vietnamese-ViVoice",
    languages=L.subset("vi"),
    license="cc-by-nc-4.0",
    optional_package="f5-tts",
    import_name="f5_tts",
    experimental=True,
    notes="Community Vietnamese checkpoint for F5-TTS.",
    supports_voice_cloning=True,
    reference_required=True,
    recommended_sample_rate=24000,
    module="providers.tts.f5",
))

DEFAULT_MODEL = "mms"
#: Legacy `tts_engine` values that are not model ids. Empty today: every engine
#: the old API accepted is still registered under the same name.
ALIASES: dict = {}
