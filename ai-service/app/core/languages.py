"""Language code registry.

The pipeline speaks ISO-639-1 internally (``en``, ``vi``, ``ja`` ...) because
that is what Whisper emits and what users pick in the UI.  Every downstream
model wants something different:

* NLLB-200      -> FLORES-200 codes   (``eng_Latn``, ``vie_Latn`` ...)
* SeamlessM4T   -> ISO-639-3          (``eng``, ``vie`` ...)
* MMS-TTS       -> ISO-639-3          (``facebook/mms-tts-vie``)
* XTTS-v2       -> its own short list (``en``, ``zh-cn`` ...)

This module is the only place that knows about those conversions.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.core.errors import UnsupportedLanguage


@dataclass(frozen=True)
class LanguageInfo:
    iso1: str          # ISO 639-1, canonical internal code
    iso3: str          # ISO 639-3, used by SeamlessM4T / MMS-TTS
    flores: str        # FLORES-200 code, used by NLLB
    name: str
    # rough syllables-per-second when read aloud; used by duration-aware translation
    speech_rate: float = 4.6


# --------------------------------------------------------------------------
# A curated table covering the languages that actually matter for a demo plus
# the long tail we can still serve through MMS-TTS. NLLB itself supports 200
# languages - any FLORES code can also be passed straight through.
# --------------------------------------------------------------------------
_LANGUAGES: tuple[LanguageInfo, ...] = (
    LanguageInfo("en", "eng", "eng_Latn", "English", 4.4),
    LanguageInfo("vi", "vie", "vie_Latn", "Vietnamese", 5.2),
    LanguageInfo("ja", "jpn", "jpn_Jpan", "Japanese", 6.5),
    LanguageInfo("ko", "kor", "kor_Hang", "Korean", 5.8),
    LanguageInfo("zh", "cmn", "zho_Hans", "Chinese (Simplified)", 5.2),
    LanguageInfo("fr", "fra", "fra_Latn", "French", 5.0),
    LanguageInfo("de", "deu", "deu_Latn", "German", 4.3),
    LanguageInfo("es", "spa", "spa_Latn", "Spanish", 5.3),
    LanguageInfo("pt", "por", "por_Latn", "Portuguese", 5.1),
    LanguageInfo("it", "ita", "ita_Latn", "Italian", 5.3),
    LanguageInfo("ru", "rus", "rus_Cyrl", "Russian", 4.6),
    LanguageInfo("nl", "nld", "nld_Latn", "Dutch", 4.5),
    LanguageInfo("pl", "pol", "pol_Latn", "Polish", 4.5),
    LanguageInfo("tr", "tur", "tur_Latn", "Turkish", 4.7),
    LanguageInfo("ar", "arb", "arb_Arab", "Arabic", 4.6),
    LanguageInfo("hi", "hin", "hin_Deva", "Hindi", 5.0),
    LanguageInfo("id", "ind", "ind_Latn", "Indonesian", 5.0),
    LanguageInfo("th", "tha", "tha_Thai", "Thai", 4.8),
    LanguageInfo("cs", "ces", "ces_Latn", "Czech", 4.6),
    LanguageInfo("hu", "hun", "hun_Latn", "Hungarian", 4.4),
    LanguageInfo("uk", "ukr", "ukr_Cyrl", "Ukrainian", 4.6),
    LanguageInfo("ro", "ron", "ron_Latn", "Romanian", 4.9),
    LanguageInfo("sv", "swe", "swe_Latn", "Swedish", 4.6),
    LanguageInfo("da", "dan", "dan_Latn", "Danish", 4.6),
    LanguageInfo("fi", "fin", "fin_Latn", "Finnish", 4.3),
    LanguageInfo("no", "nob", "nob_Latn", "Norwegian", 4.6),
    LanguageInfo("el", "ell", "ell_Grek", "Greek", 5.0),
    LanguageInfo("he", "heb", "heb_Hebr", "Hebrew", 4.6),
    LanguageInfo("ms", "zsm", "zsm_Latn", "Malay", 5.0),
    LanguageInfo("fa", "pes", "pes_Arab", "Persian", 4.7),
    LanguageInfo("bn", "ben", "ben_Beng", "Bengali", 4.8),
    LanguageInfo("ta", "tam", "tam_Taml", "Tamil", 4.6),
    LanguageInfo("te", "tel", "tel_Telu", "Telugu", 4.6),
    LanguageInfo("ur", "urd", "urd_Arab", "Urdu", 4.7),
    LanguageInfo("sw", "swh", "swh_Latn", "Swahili", 4.8),
    LanguageInfo("tl", "tgl", "tgl_Latn", "Tagalog", 5.0),
    LanguageInfo("km", "khm", "khm_Khmr", "Khmer", 4.7),
    LanguageInfo("lo", "lao", "lao_Laoo", "Lao", 4.7),
    LanguageInfo("my", "mya", "mya_Mymr", "Burmese", 4.6),
    LanguageInfo("bg", "bul", "bul_Cyrl", "Bulgarian", 4.6),
    LanguageInfo("hr", "hrv", "hrv_Latn", "Croatian", 4.6),
    LanguageInfo("sr", "srp", "srp_Cyrl", "Serbian", 4.6),
    LanguageInfo("sk", "slk", "slk_Latn", "Slovak", 4.6),
    LanguageInfo("sl", "slv", "slv_Latn", "Slovenian", 4.6),
    LanguageInfo("ca", "cat", "cat_Latn", "Catalan", 5.1),
    LanguageInfo("hy", "hye", "hye_Armn", "Armenian", 4.5),
    LanguageInfo("ka", "kat", "kat_Geor", "Georgian", 4.4),
    LanguageInfo("ne", "npi", "npi_Deva", "Nepali", 4.8),
    LanguageInfo("si", "sin", "sin_Sinh", "Sinhala", 4.6),
    LanguageInfo("mn", "khk", "khk_Cyrl", "Mongolian", 4.4),
)

BY_ISO1: dict[str, LanguageInfo] = {lang.iso1: lang for lang in _LANGUAGES}
BY_ISO3: dict[str, LanguageInfo] = {lang.iso3: lang for lang in _LANGUAGES}
BY_FLORES: dict[str, LanguageInfo] = {lang.flores: lang for lang in _LANGUAGES}

# Whisper occasionally reports these; normalise them.
_ALIASES = {
    "zh-cn": "zh", "zh_cn": "zh", "zh-hans": "zh", "cmn": "zh",
    "zh-tw": "zh", "yue": "zh",
    "pt-br": "pt", "pt_br": "pt",
    "nb": "no", "nn": "no",
    "iw": "he", "in": "id", "jw": "id",
    "vn": "vi",
}

# XTTS-v2 (voice cloning) officially supports exactly these.
XTTS_LANGUAGES: dict[str, str] = {
    "en": "en", "es": "es", "fr": "fr", "de": "de", "it": "it", "pt": "pt",
    "pl": "pl", "tr": "tr", "ru": "ru", "nl": "nl", "cs": "cs", "ar": "ar",
    "zh": "zh-cn", "hu": "hu", "ko": "ko", "ja": "ja", "hi": "hi",
}


def normalize(code: str | None) -> str | None:
    """Normalise any incoming language code to canonical ISO-639-1."""
    if not code:
        return None
    raw = code.strip().lower().replace("_", "-")
    if raw in _ALIASES:
        return _ALIASES[raw]
    if raw in BY_ISO1:
        return raw
    if raw in BY_ISO3:
        return BY_ISO3[raw].iso1
    # FLORES style: eng_Latn
    flores_key = code.strip()
    if flores_key in BY_FLORES:
        return BY_FLORES[flores_key].iso1
    if "-" in raw:  # en-US -> en
        head = raw.split("-", 1)[0]
        if head in BY_ISO1:
            return head
    return None


def get(code: str) -> LanguageInfo:
    iso1 = normalize(code)
    if iso1 is None or iso1 not in BY_ISO1:
        raise UnsupportedLanguage(
            f"Language '{code}' is not in the registry.",
            details={"requested": code, "supported": sorted(BY_ISO1)},
        )
    return BY_ISO1[iso1]


def to_flores(code: str) -> str:
    """FLORES-200 code for NLLB. Accepts FLORES codes verbatim as an escape hatch."""
    if code in BY_FLORES:
        return code
    if "_" in code and len(code.split("_")[0]) == 3:
        return code  # trust an explicit FLORES code we do not have in the table
    return get(code).flores


def to_iso3(code: str) -> str:
    return get(code).iso3


def to_xtts(code: str) -> str | None:
    iso1 = normalize(code)
    return XTTS_LANGUAGES.get(iso1) if iso1 else None


def supports_voice_cloning(code: str) -> bool:
    return to_xtts(code) is not None


def speech_rate(code: str) -> float:
    try:
        return get(code).speech_rate
    except UnsupportedLanguage:
        return 4.6


def catalog() -> list[dict]:
    """Serialisable list for the frontend language pickers."""
    return [
        {
            "code": lang.iso1,
            "iso3": lang.iso3,
            "flores": lang.flores,
            "name": lang.name,
            "voice_cloning": lang.iso1 in XTTS_LANGUAGES,
        }
        for lang in sorted(_LANGUAGES, key=lambda x: x.name)
    ]
