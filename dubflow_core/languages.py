"""The one language table.

ISO-639-1 is the public application code. Models want other alphabets: MMS and
SeamlessM4T use ISO-639-3, NLLB uses FLORES-200 codes. Providers translate from
here rather than each keeping their own map.
"""
from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional


class Language(NamedTuple):
    code: str      #: ISO-639-1, the code every API in this project speaks
    name: str      #: English display name
    iso3: str      #: ISO-639-3, used by MMS and SeamlessM4T
    flores: str    #: FLORES-200 code, used by NLLB


_ROWS = (
    ("en", "English", "eng", "eng_Latn"),
    ("vi", "Vietnamese", "vie", "vie_Latn"),
    ("ja", "Japanese", "jpn", "jpn_Jpan"),
    ("ko", "Korean", "kor", "kor_Hang"),
    ("zh", "Chinese", "cmn", "zho_Hans"),
    ("fr", "French", "fra", "fra_Latn"),
    ("de", "German", "deu", "deu_Latn"),
    ("es", "Spanish", "spa", "spa_Latn"),
    ("pt", "Portuguese", "por", "por_Latn"),
    ("it", "Italian", "ita", "ita_Latn"),
    ("ru", "Russian", "rus", "rus_Cyrl"),
    ("nl", "Dutch", "nld", "nld_Latn"),
    ("pl", "Polish", "pol", "pol_Latn"),
    ("tr", "Turkish", "tur", "tur_Latn"),
    ("ar", "Arabic", "ara", "arb_Arab"),
    ("hi", "Hindi", "hin", "hin_Deva"),
    ("id", "Indonesian", "ind", "ind_Latn"),
    ("th", "Thai", "tha", "tha_Thai"),
    ("cs", "Czech", "ces", "ces_Latn"),
    ("hu", "Hungarian", "hun", "hun_Latn"),
    ("uk", "Ukrainian", "ukr", "ukr_Cyrl"),
    ("ro", "Romanian", "ron", "ron_Latn"),
    ("sv", "Swedish", "swe", "swe_Latn"),
    ("da", "Danish", "dan", "dan_Latn"),
    ("fi", "Finnish", "fin", "fin_Latn"),
    ("no", "Norwegian", "nob", "nob_Latn"),
    ("el", "Greek", "ell", "ell_Grek"),
    ("he", "Hebrew", "heb", "heb_Hebr"),
    ("ms", "Malay", "zsm", "zsm_Latn"),
    ("fa", "Persian", "pes", "pes_Arab"),
    ("bn", "Bengali", "ben", "ben_Beng"),
    ("ta", "Tamil", "tam", "tam_Taml"),
    ("te", "Telugu", "tel", "tel_Telu"),
    ("ur", "Urdu", "urd", "urd_Arab"),
    ("sw", "Swahili", "swh", "swh_Latn"),
    ("tl", "Tagalog", "tgl", "tgl_Latn"),
    ("km", "Khmer", "khm", "khm_Khmr"),
    ("lo", "Lao", "lao", "lao_Laoo"),
    ("my", "Burmese", "mya", "mya_Mymr"),
    ("bg", "Bulgarian", "bul", "bul_Cyrl"),
    ("hr", "Croatian", "hrv", "hrv_Latn"),
    ("sr", "Serbian", "srp", "srp_Cyrl"),
    ("sk", "Slovak", "slk", "slk_Latn"),
    ("sl", "Slovenian", "slv", "slv_Latn"),
    ("ca", "Catalan", "cat", "cat_Latn"),
    ("hy", "Armenian", "hye", "hye_Armn"),
    ("ka", "Georgian", "kat", "kat_Geor"),
    ("ne", "Nepali", "npi", "npi_Deva"),
    ("si", "Sinhala", "sin", "sin_Sinh"),
    ("mn", "Mongolian", "khk", "khk_Cyrl"),
)

LANGUAGES: Dict[str, Language] = {row[0]: Language(*row) for row in _ROWS}
CODES = tuple(LANGUAGES)
#: Every code, for providers whose coverage is the whole table.
ALL = CODES


def normalise(code: Optional[str]) -> Optional[str]:
    """Lower-case and trim; "" and "auto" both mean "detect it"."""
    if code is None:
        return None
    cleaned = code.strip().lower()
    return None if cleaned in {"", "auto"} else cleaned


def is_supported(code: Optional[str]) -> bool:
    return code in LANGUAGES


def name_of(code: str) -> str:
    return LANGUAGES[code].name if code in LANGUAGES else code


def iso3(code: str) -> str:
    return LANGUAGES[code].iso3


def flores(code: str) -> str:
    return LANGUAGES[code].flores


def subset(*codes: str) -> tuple:
    """Build a provider's language tuple, rejecting codes outside the table.

    A typo in a provider's language list would otherwise become a silent claim
    of support for a language the pipeline cannot even name.
    """
    unknown = [code for code in codes if code not in LANGUAGES]
    if unknown:
        raise ValueError(f"Not application language codes: {sorted(unknown)}")
    return tuple(dict.fromkeys(codes))


def exclude(*codes: str) -> tuple:
    """Every application language except the ones named."""
    unknown = [code for code in codes if code not in LANGUAGES]
    if unknown:
        raise ValueError(f"Not application language codes: {sorted(unknown)}")
    removed = set(codes)
    return tuple(code for code in CODES if code not in removed)


def listing() -> List[dict]:
    return [{"code": row.code, "name": row.name} for row in LANGUAGES.values()]
