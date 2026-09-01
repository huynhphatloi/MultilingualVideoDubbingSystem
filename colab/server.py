"""Google Colab AI backend for transcription, translation, and speech."""
from __future__ import annotations

import gc
import io
import os
import shutil
import tempfile
import threading
from pathlib import Path

import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, Header, HTTPException, Response, UploadFile
from faster_whisper import WhisperModel
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, VitsModel

AUTH_TOKEN = os.getenv("AUTH_TOKEN", "")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
TRANSLATION_MODEL = os.getenv(
    "TRANSLATION_MODEL", "facebook/nllb-200-distilled-600M"
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ISO-639-1 is the public API. MMS uses ISO-639-3 and NLLB uses FLORES codes.
LANGUAGES = {
    "en": ("English", "eng", "eng_Latn"),
    "vi": ("Vietnamese", "vie", "vie_Latn"),
    "ja": ("Japanese", "jpn", "jpn_Jpan"),
    "ko": ("Korean", "kor", "kor_Hang"),
    "zh": ("Chinese", "cmn", "zho_Hans"),
    "fr": ("French", "fra", "fra_Latn"),
    "de": ("German", "deu", "deu_Latn"),
    "es": ("Spanish", "spa", "spa_Latn"),
    "pt": ("Portuguese", "por", "por_Latn"),
    "it": ("Italian", "ita", "ita_Latn"),
    "ru": ("Russian", "rus", "rus_Cyrl"),
    "nl": ("Dutch", "nld", "nld_Latn"),
    "pl": ("Polish", "pol", "pol_Latn"),
    "tr": ("Turkish", "tur", "tur_Latn"),
    "ar": ("Arabic", "ara", "arb_Arab"),
    "hi": ("Hindi", "hin", "hin_Deva"),
    "id": ("Indonesian", "ind", "ind_Latn"),
    "th": ("Thai", "tha", "tha_Thai"),
    "cs": ("Czech", "ces", "ces_Latn"),
    "hu": ("Hungarian", "hun", "hun_Latn"),
    "uk": ("Ukrainian", "ukr", "ukr_Cyrl"),
    "ro": ("Romanian", "ron", "ron_Latn"),
    "sv": ("Swedish", "swe", "swe_Latn"),
    "da": ("Danish", "dan", "dan_Latn"),
    "fi": ("Finnish", "fin", "fin_Latn"),
    "no": ("Norwegian", "nob", "nob_Latn"),
    "el": ("Greek", "ell", "ell_Grek"),
    "he": ("Hebrew", "heb", "heb_Hebr"),
    "ms": ("Malay", "zsm", "zsm_Latn"),
    "fa": ("Persian", "pes", "pes_Arab"),
    "bn": ("Bengali", "ben", "ben_Beng"),
    "ta": ("Tamil", "tam", "tam_Taml"),
    "te": ("Telugu", "tel", "tel_Telu"),
    "ur": ("Urdu", "urd", "urd_Arab"),
    "sw": ("Swahili", "swh", "swh_Latn"),
    "tl": ("Tagalog", "tgl", "tgl_Latn"),
    "km": ("Khmer", "khm", "khm_Khmr"),
    "lo": ("Lao", "lao", "lao_Laoo"),
    "my": ("Burmese", "mya", "mya_Mymr"),
    "bg": ("Bulgarian", "bul", "bul_Cyrl"),
    "hr": ("Croatian", "hrv", "hrv_Latn"),
    "sr": ("Serbian", "srp", "srp_Cyrl"),
    "sk": ("Slovak", "slk", "slk_Latn"),
    "sl": ("Slovenian", "slv", "slv_Latn"),
    "ca": ("Catalan", "cat", "cat_Latn"),
    "hy": ("Armenian", "hye", "hye_Armn"),
    "ka": ("Georgian", "kat", "kat_Geor"),
    "ne": ("Nepali", "npi", "npi_Deva"),
    "si": ("Sinhala", "sin", "sin_Sinh"),
    "mn": ("Mongolian", "khk", "khk_Cyrl"),
}

app = FastAPI(title="DubFlow Colab AI", version="2.0")
_lock = threading.Lock()
_whisper = None
_translation = None
_tts_tokenizer = None
_tts_model = None
_tts_language: str | None = None


class TranslationRequest(BaseModel):
    source_language: str
    target_language: str
    texts: list[str]


def _authorize(authorization: str | None) -> None:
    if AUTH_TOKEN and authorization != f"Bearer {AUTH_TOKEN}":
        raise HTTPException(status_code=401, detail="Invalid bearer token")


def _validate_language(language: str) -> str:
    language = language.strip().lower()
    if language not in LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Unsupported language '{language}'")
    return language


def _load_whisper():  # noqa: ANN202
    global _whisper
    if _whisper is None:
        compute_type = "float16" if DEVICE == "cuda" else "int8"
        _whisper = WhisperModel(
            WHISPER_MODEL, device=DEVICE, compute_type=compute_type
        )
    return _whisper


def _load_translation():  # noqa: ANN202
    global _translation
    if _translation is None:
        tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL)
        model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL)
        _translation = (tokenizer, model.to(DEVICE).eval())
    return _translation


def _load_tts(language: str):  # noqa: ANN202
    """Keep only the selected TTS language model resident in memory."""
    global _tts_language, _tts_model, _tts_tokenizer
    if _tts_language == language and _tts_model is not None:
        return _tts_tokenizer, _tts_model

    _tts_tokenizer = None
    _tts_model = None
    _tts_language = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    repository = f"facebook/mms-tts-{LANGUAGES[language][1]}"
    _tts_tokenizer = AutoTokenizer.from_pretrained(repository)
    _tts_model = VitsModel.from_pretrained(repository).to(DEVICE).eval()
    _tts_language = language
    return _tts_tokenizer, _tts_model


@app.get("/health")
def health() -> dict:
    return {
        "status": "ready",
        "device": DEVICE,
        "services": ["transcription", "translation", "speech"],
        "whisper_model": WHISPER_MODEL,
        "translation_model": TRANSLATION_MODEL,
        "tts_model": f"facebook/mms-tts-{LANGUAGES[_tts_language][1]}"
        if _tts_language
        else None,
        "languages": [
            {"code": code, "name": values[0]}
            for code, values in LANGUAGES.items()
        ],
    }


@app.post("/transcribe")
def transcribe(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    authorization: str | None = Header(None),
) -> dict:
    _authorize(authorization)
    requested = language.strip().lower()
    source = None if requested in {"", "auto"} else _validate_language(requested)
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
            shutil.copyfileobj(file.file, temporary)
            temporary_path = temporary.name
        with _lock:
            raw_segments, info = _load_whisper().transcribe(
                temporary_path,
                language=source,
                vad_filter=True,
                beam_size=5,
            )
            segments = [
                {
                    "id": index,
                    "start": round(float(segment.start), 3),
                    "end": round(float(segment.end), 3),
                    "source_text": segment.text.strip(),
                }
                for index, segment in enumerate(raw_segments)
                if segment.text.strip()
            ]
        detected = source or str(info.language)
        _validate_language(detected)
        if not segments:
            raise HTTPException(status_code=422, detail="Whisper found no speech")
        return {"source_language": detected, "segments": segments}
    finally:
        if temporary_path:
            Path(temporary_path).unlink(missing_ok=True)


@app.post("/translate")
def translate(
    request: TranslationRequest,
    authorization: str | None = Header(None),
) -> dict:
    _authorize(authorization)
    source = _validate_language(request.source_language)
    target = _validate_language(request.target_language)
    if not request.texts or len(request.texts) > 500:
        raise HTTPException(status_code=400, detail="texts must contain 1 to 500 items")
    if source == target:
        return {"translations": request.texts}

    with _lock:
        tokenizer, model = _load_translation()
        tokenizer.src_lang = LANGUAGES[source][2]
        encoded = tokenizer(
            request.texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        encoded = {name: value.to(DEVICE) for name, value in encoded.items()}
        target_token = tokenizer.convert_tokens_to_ids(LANGUAGES[target][2])
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                forced_bos_token_id=target_token,
                max_new_tokens=256,
                num_beams=4,
            )
        translations = tokenizer.batch_decode(output, skip_special_tokens=True)
    return {"translations": [text.strip() for text in translations]}


@app.post("/synthesize")
def synthesize(
    text: str = Form(...),
    language: str = Form("vi"),
    speed: float = Form(1.0),
    authorization: str | None = Header(None),
) -> Response:
    _authorize(authorization)
    language = _validate_language(language)
    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    if len(text) > 2000:
        raise HTTPException(status_code=400, detail="Text must be at most 2000 characters")
    if not 0.5 <= speed <= 2.0:
        raise HTTPException(status_code=400, detail="Speed must be between 0.5 and 2.0")

    with _lock:
        tokenizer, model = _load_tts(language)
        encoded = tokenizer(text, return_tensors="pt")
        encoded = {name: value.to(DEVICE) for name, value in encoded.items()}
        model.speaking_rate = speed
        with torch.inference_mode():
            waveform = model(**encoded).waveform.squeeze().float().cpu().numpy()
        sample_rate = int(model.config.sampling_rate)

    output = io.BytesIO()
    sf.write(output, waveform, sample_rate, format="WAV", subtype="PCM_16")
    repository = f"facebook/mms-tts-{LANGUAGES[language][1]}"
    return Response(
        content=output.getvalue(),
        media_type="audio/wav",
        headers={"X-TTS-Model": repository, "X-TTS-Sample-Rate": str(sample_rate)},
    )
