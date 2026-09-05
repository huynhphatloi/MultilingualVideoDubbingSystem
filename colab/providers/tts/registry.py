"""Speech engines this build knows about.

Every language list below was checked against the engine itself, not assumed:

* MMS-TTS - the `facebook/mms-tts-<iso3>` repositories that actually exist. 34
  of the 50 application languages have one; Japanese, Chinese, Italian, Czech,
  Danish and Norwegian are among those that do not.
* Edge - the locale list the Microsoft voice service returns. Every application
  language except Armenian has a voice; Norwegian is `nb` and Tagalog is `fil`,
  which providers.tts.edge maps.
* Piper - the language folders in rhasspy/piper-voices.
* XTTS-v2 - the 17 languages Coqui documents. No Vietnamese.
* viXTTS, F5 Vietnamese - single-language Vietnamese checkpoints.
* F5-TTS base - English and Chinese.
* Chatterbox - the SUPPORTED_LANGUAGES table in its own source: 23 languages,
  no Vietnamese.
* Kokoro - the 9 pipeline language codes, which map to 8 application languages.
* OpenVoice V2 - "English, Spanish, French, Chinese, Japanese and Korean".
* CosyVoice 2 - "Chinese, English, Japanese, Korean, German, Spanish, French,
  Italian, Russian".
* VieNeu-TTS - Vietnamese.
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
    ProviderSpec("piper", "tts", "Piper",
                 "https://huggingface.co/rhasspy/piper-voices",
                 "Fast local ONNX voices."),
    ProviderSpec("xtts", "tts", "Coqui XTTS",
                 "https://huggingface.co/coqui/XTTS-v2",
                 "Voice cloning. Coqui's own weights are under the CPML licence."),
    ProviderSpec("f5", "tts", "F5-TTS",
                 "https://github.com/SWivid/F5-TTS",
                 "Flow-matching voice cloning."),
    ProviderSpec("chatterbox", "tts", "Chatterbox",
                 "https://huggingface.co/ResembleAI/chatterbox",
                 "MIT-licensed zero-shot cloning, 23 languages."),
    ProviderSpec("kokoro", "tts", "Kokoro",
                 "https://huggingface.co/hexgrad/Kokoro-82M",
                 "82M parameters, Apache-2.0, stock voices only."),
    ProviderSpec("openvoice", "tts", "OpenVoice V2",
                 "https://huggingface.co/myshell-ai/OpenVoiceV2",
                 "Tone-colour conversion on top of MeloTTS."),
    ProviderSpec("cosyvoice", "tts", "CosyVoice 2",
                 "https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B",
                 "Zero-shot cloning; installed from its own repository."),
    ProviderSpec("vieneu", "tts", "VieNeu-TTS",
                 "https://huggingface.co/pnnbao-ump/VieNeu-TTS",
                 "Vietnamese cloning that runs on CPU."),
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
    id="piper",
    task="tts",
    provider="piper",
    display_name="Piper",
    repo_id="rhasspy/piper-voices",
    #: The language folders published in the voice repository.
    languages=L.subset(
        "en", "vi", "ja", "ko", "zh", "fr", "de", "es", "pt", "it", "ru", "nl",
        "pl", "tr", "ar", "hi", "id", "th", "cs", "hu", "uk", "ro", "sv", "da",
        "fi", "no", "el", "he", "fa", "bn", "te", "ur", "sw", "bg", "sr", "sk",
        "sl", "ca", "hy", "ka", "ne",
    ),
    multilingual=True,
    license="mit",
    optional_package="piper-tts",
    import_name="piper",
    notes="Runs on CPU in real time. Voice quality is below the cloning engines.",
    supports_multispeaker=True,
    recommended_sample_rate=22050,
    module="providers.tts.piper",
))

REGISTRY.add(ModelSpec(
    id="xtts_v2",
    task="tts",
    provider="xtts",
    display_name="XTTS v2",
    repo_id="coqui/XTTS-v2",
    #: The 17 languages Coqui documents. Vietnamese is not one of them.
    languages=L.subset(
        "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar",
        "zh", "hu", "ko", "ja", "hi",
    ),
    multilingual=True,
    license="Coqui Public Model License (non-commercial)",
    optional_package="coqui-tts",
    import_name="TTS",
    notes="No Vietnamese. Use viXTTS or the F5 Vietnamese checkpoint for that.",
    supports_voice_cloning=True,
    reference_required=True,
    recommended_sample_rate=24000,
    module="providers.tts.xtts",
))

REGISTRY.add(ModelSpec(
    id="vixtts",
    task="tts",
    provider="xtts",
    display_name="viXTTS (Vietnamese)",
    repo_id="capleaf/viXTTS",
    languages=L.subset("vi"),
    license="Coqui Public Model License (non-commercial)",
    optional_package="coqui-tts",
    import_name="TTS",
    experimental=True,
    notes="Community fine-tune of XTTS v2 on Vietnamese speech.",
    supports_voice_cloning=True,
    reference_required=True,
    recommended_sample_rate=24000,
    module="providers.tts.xtts",
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

REGISTRY.add(ModelSpec(
    id="chatterbox",
    task="tts",
    provider="chatterbox",
    display_name="Chatterbox Multilingual",
    repo_id="ResembleAI/chatterbox",
    #: SUPPORTED_LANGUAGES in chatterbox's own source. No Vietnamese.
    languages=L.subset(
        "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja",
        "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh",
    ),
    multilingual=True,
    license="mit",
    optional_package="chatterbox-tts",
    import_name="chatterbox",
    notes="Clones from a reference clip when given one and uses its built-in "
          "voice otherwise. Watermarks its output, by design.",
    supports_voice_cloning=True,
    reference_required=False,
    recommended_sample_rate=24000,
    module="providers.tts.chatterbox",
))

REGISTRY.add(ModelSpec(
    id="chatterbox_en",
    task="tts",
    provider="chatterbox",
    display_name="Chatterbox English",
    repo_id="ResembleAI/chatterbox",
    languages=L.subset("en"),
    license="mit",
    optional_package="chatterbox-tts",
    import_name="chatterbox",
    notes="The English-only checkpoint; slightly better English than the "
          "multilingual one.",
    supports_voice_cloning=True,
    reference_required=False,
    recommended_sample_rate=24000,
    module="providers.tts.chatterbox",
))

REGISTRY.add(ModelSpec(
    id="kokoro",
    task="tts",
    provider="kokoro",
    display_name="Kokoro 82M",
    repo_id="hexgrad/Kokoro-82M",
    #: The nine pipeline codes cover eight application languages (American and
    #: British English are two codes for one).
    languages=L.subset("en", "es", "fr", "hi", "it", "ja", "pt", "zh"),
    multilingual=True,
    license="apache-2.0",
    optional_package="kokoro",
    import_name="kokoro",
    extra_requirements=("espeak-ng (apt) for es/fr/hi/it/pt",
                        "misaki[ja] for Japanese", "misaki[zh] for Chinese"),
    notes="Stock voices only - it cannot clone, so voice cloning must be off.",
    supports_multispeaker=True,
    speed_control="native",
    recommended_sample_rate=24000,
    module="providers.tts.kokoro",
))

REGISTRY.add(ModelSpec(
    id="openvoice_v2",
    task="tts",
    provider="openvoice",
    display_name="OpenVoice V2",
    repo_id="myshell-ai/OpenVoiceV2",
    languages=L.subset("en", "es", "fr", "zh", "ja", "ko"),
    multilingual=True,
    license="mit",
    optional_package="git+https://github.com/myshell-ai/OpenVoice.git",
    import_name="openvoice",
    extra_requirements=(
        "git+https://github.com/myshell-ai/MeloTTS.git",
        "python -m unidic download",
    ),
    experimental=True,
    notes="Not one model but two: MeloTTS speaks the line and OpenVoice moves "
          "it onto the reference speaker's tone colour. Both have to be "
          "installed, and MeloTTS pins several of its own dependencies.",
    supports_voice_cloning=True,
    reference_required=True,
    speed_control="native",
    recommended_sample_rate=24000,
    module="providers.tts.openvoice",
))

REGISTRY.add(ModelSpec(
    id="cosyvoice2",
    task="tts",
    provider="cosyvoice",
    display_name="CosyVoice 2 0.5B",
    repo_id="FunAudioLLM/CosyVoice2-0.5B",
    languages=L.subset("zh", "en", "ja", "ko", "de", "es", "fr", "it", "ru"),
    multilingual=True,
    license="apache-2.0",
    optional_package="cosyvoice (git clone, see the notebook)",
    import_name="cosyvoice",
    extra_requirements=(
        "git clone --recursive https://github.com/FunAudioLLM/CosyVoice",
        "COSYVOICE_ROOT must point at that checkout",
    ),
    experimental=True,
    notes="Not on PyPI: it is installed from its repository and needs "
          "third_party/Matcha-TTS on sys.path. Zero-shot cloning wants the "
          "transcript of the reference clip as well as the audio.",
    supports_voice_cloning=True,
    reference_required=True,
    reference_text_required=True,
    recommended_sample_rate=24000,
    module="providers.tts.cosyvoice",
))

REGISTRY.add(ModelSpec(
    id="vieneu",
    task="tts",
    provider="vieneu",
    display_name="VieNeu-TTS (Vietnamese)",
    repo_id="pnnbao-ump/VieNeu-TTS-v3-Turbo",
    languages=L.subset("vi"),
    license="apache-2.0",
    optional_package="vieneu",
    import_name="vieneu",
    experimental=True,
    notes="Community Vietnamese model built on NeuTTS Air. Clones from three "
          "to eight seconds of reference audio and needs no reference text.",
    supports_voice_cloning=True,
    reference_required=True,
    streaming=True,
    recommended_sample_rate=48000,
    module="providers.tts.vieneu",
))

DEFAULT_MODEL = "mms"
#: Legacy `tts_engine` values that are not model ids. Empty today: every engine
#: the old API accepted is still registered under the same name.
ALIASES: dict = {}
