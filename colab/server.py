"""Google Colab AI backend for transcription, translation, and speech.

Every stage picks its model per request. Only the base install is assumed to be
present; an engine whose package is missing fails with the pip command that
fixes it instead of a bare ImportError.
"""
from __future__ import annotations

import gc
import importlib.util
import io
import json
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, Header, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse
from faster_whisper import WhisperModel
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, VitsModel

AUTH_TOKEN = os.getenv("AUTH_TOKEN", "")
#: Only the fallback when a request omits the model; the form always sends one.
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
NLLB_MODEL = os.getenv("TRANSLATION_MODEL", "facebook/nllb-200-distilled-600M")
SEAMLESS_MODEL = os.getenv("SEAMLESS_MODEL", "facebook/hf-seamless-m4t-medium")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#: Checkpoints faster-whisper resolves by name. Mirrors the local API's list.
WHISPER_MODELS = {
    "tiny", "tiny.en", "base", "base.en", "small", "small.en",
    "medium", "medium.en", "large-v2", "large-v3", "large-v3-turbo",
    "distil-small.en", "distil-medium.en", "distil-large-v3",
}

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

app = FastAPI(title="DubFlow Colab AI", version="3.0")
_lock = threading.Lock()

# One model of each kind stays resident. Switching evicts the previous one so a
# free Colab GPU is not asked to hold two large checkpoints at once.
_whisper = None
_whisper_name: str | None = None
_translation = None
_translation_engine: str | None = None
_tts_cache: dict = {}
_tts_key: str | None = None

_REFERENCE_DIR = Path(tempfile.gettempdir()) / "dubflow-references"
_REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
_REFERENCE_LIMIT = 8


class TranslationRequest(BaseModel):
    source_language: str
    target_language: str
    texts: list[str]
    engine: str = "nllb"


def _authorize(authorization: str | None) -> None:
    if AUTH_TOKEN and authorization != f"Bearer {AUTH_TOKEN}":
        raise HTTPException(status_code=401, detail="Invalid bearer token")


def _validate_language(language: str) -> str:
    language = language.strip().lower()
    if language not in LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Unsupported language '{language}'")
    return language


def _free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _require(package: str, pip_name: str) -> None:
    """Fail with the pip command rather than a bare ImportError."""
    if importlib.util.find_spec(package) is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"This engine needs the '{pip_name}' package, which the notebook's "
                f"base install does not include. Run `pip install {pip_name}` in a "
                f"Colab cell, then restart the API cell."
            ),
        )


def _ffmpeg(source: Path, target: Path, speed: float = 1.0) -> None:
    """Normalise any engine's output to mono 24 kHz WAV, applying speed."""
    filters = ["aresample=24000", "aformat=channel_layouts=mono"]
    # atempo only accepts 0.5-2.0 per stage, which covers the validated range.
    if abs(speed - 1.0) > 1e-3:
        filters.insert(0, f"atempo={speed:.3f}")
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(source), "-af", ",".join(filters),
            "-c:a", "pcm_s16le", str(target),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        detail = (result.stderr or "ffmpeg failed")[-500:]
        raise HTTPException(status_code=500, detail=f"Audio conversion failed: {detail}")


def _write_wave(waveform, sample_rate: int, target: Path, speed: float = 1.0) -> None:
    if abs(speed - 1.0) <= 1e-3:
        sf.write(str(target), waveform, sample_rate, format="WAV", subtype="PCM_16")
        return
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as raw:
        raw_path = Path(raw.name)
    try:
        sf.write(str(raw_path), waveform, sample_rate, format="WAV", subtype="PCM_16")
        _ffmpeg(raw_path, target, speed)
    finally:
        raw_path.unlink(missing_ok=True)


# --------------------------------------------------------------------------
# Transcription
# --------------------------------------------------------------------------
def _load_whisper(name: str):  # noqa: ANN202
    global _whisper, _whisper_name
    if name not in WHISPER_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported Whisper model '{name}' (pick one of {sorted(WHISPER_MODELS)})",
        )
    if _whisper_name == name and _whisper is not None:
        return _whisper

    _whisper = None
    _whisper_name = None
    _free_memory()
    compute_type = "float16" if DEVICE == "cuda" else "int8"
    _whisper = WhisperModel(name, device=DEVICE, compute_type=compute_type)
    _whisper_name = name
    return _whisper


# --------------------------------------------------------------------------
# Translation
# --------------------------------------------------------------------------
def _load_translation(engine: str):  # noqa: ANN202
    global _translation, _translation_engine
    if _translation_engine == engine and _translation is not None:
        return _translation

    _translation = None
    _translation_engine = None
    _free_memory()

    if engine == "nllb":
        tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL)
        model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL)
        _translation = (tokenizer, model.to(DEVICE).eval(), NLLB_MODEL)
    elif engine == "seamless":
        from transformers import AutoProcessor, SeamlessM4TModel

        processor = AutoProcessor.from_pretrained(SEAMLESS_MODEL)
        model = SeamlessM4TModel.from_pretrained(SEAMLESS_MODEL)
        _translation = (processor, model.to(DEVICE).eval(), SEAMLESS_MODEL)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported translation engine '{engine}' (pick nllb or seamless)",
        )
    _translation_engine = engine
    return _translation


def _translate_nllb(texts: list[str], source: str, target: str) -> list[str]:
    tokenizer, model, _ = _load_translation("nllb")
    tokenizer.src_lang = LANGUAGES[source][2]
    encoded = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=512
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
    return tokenizer.batch_decode(output, skip_special_tokens=True)


def _translate_seamless(texts: list[str], source: str, target: str) -> list[str]:
    processor, model, _ = _load_translation("seamless")
    encoded = processor(
        text=texts,
        src_lang=LANGUAGES[source][1],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    encoded = {name: value.to(DEVICE) for name, value in encoded.items()}
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            tgt_lang=LANGUAGES[target][1],
            generate_speech=False,
            num_beams=4,
        )
    # Text-only generation returns a wrapper whose first element is the tokens.
    sequences = getattr(output, "sequences", None)
    if sequences is None:
        sequences = output[0] if isinstance(output, (tuple, list)) else output
    return processor.batch_decode(sequences, skip_special_tokens=True)


TRANSLATION_ENGINES = {"nllb": _translate_nllb, "seamless": _translate_seamless}


# --------------------------------------------------------------------------
# Speech
# --------------------------------------------------------------------------
#: XTTS speaks these; "zh" has to travel as "zh-cn" and Vietnamese is absent.
_XTTS_LANGUAGES = {
    "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru",
    "nl", "cs", "ar", "zh", "hu", "ko", "ja", "hi",
}


def _tts_slot(key: str, builder):  # noqa: ANN001, ANN202
    """Hold one speech model at a time, evicting the previous engine first."""
    global _tts_cache, _tts_key
    if _tts_key == key and _tts_cache:
        return _tts_cache
    _tts_cache = {}
    _tts_key = None
    _free_memory()
    _tts_cache = builder()
    _tts_key = key
    return _tts_cache


def _needs_reference(reference: Path | None, engine: str) -> Path:
    if reference is None or not reference.exists():
        raise HTTPException(
            status_code=400,
            detail=(
                f"The '{engine}' engine clones a voice and needs a reference sample. "
                f"Upload one to /reference first and pass its reference_id."
            ),
        )
    return reference


def _tts_mms(text: str, language: str, speed: float, reference, out: Path) -> str:
    repository = f"facebook/mms-tts-{LANGUAGES[language][1]}"

    def build() -> dict:
        tokenizer = AutoTokenizer.from_pretrained(repository)
        model = VitsModel.from_pretrained(repository).to(DEVICE).eval()
        return {"tokenizer": tokenizer, "model": model}

    slot = _tts_slot(f"mms:{language}", build)
    encoded = slot["tokenizer"](text, return_tensors="pt")
    encoded = {name: value.to(DEVICE) for name, value in encoded.items()}
    # MMS varies rate natively, so no atempo pass is needed afterwards.
    slot["model"].speaking_rate = speed
    with torch.inference_mode():
        waveform = slot["model"](**encoded).waveform.squeeze().float().cpu().numpy()
    _write_wave(waveform, int(slot["model"].config.sampling_rate), out)
    return repository


def _tts_edge(text: str, language: str, speed: float, reference, out: Path) -> str:
    _require("edge_tts", "edge-tts")
    import asyncio

    import edge_tts

    def build() -> dict:
        # Ask the service for its catalogue rather than pinning voice names that
        # Microsoft renames without notice.
        voices = asyncio.run(edge_tts.list_voices())
        return {"voices": voices}

    slot = _tts_slot("edge", build)
    matches = [
        voice for voice in slot["voices"]
        if str(voice.get("ShortName", "")).lower().startswith(f"{language}-")
    ]
    if not matches:
        raise HTTPException(
            status_code=400,
            detail=f"Edge TTS has no voice for '{language}'. Use the 'mms' engine instead.",
        )
    female = [v for v in matches if str(v.get("Gender", "")).lower() == "female"]
    voice = (female or matches)[0]["ShortName"]

    percent = int(round((speed - 1.0) * 100))
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as raw:
        raw_path = Path(raw.name)
    try:
        communicate = edge_tts.Communicate(text, voice, rate=f"{percent:+d}%")
        asyncio.run(communicate.save(str(raw_path)))
        if not raw_path.exists() or raw_path.stat().st_size == 0:
            raise HTTPException(status_code=502, detail="Edge TTS returned no audio")
        # Speed already applied through `rate`, so convert without atempo.
        _ffmpeg(raw_path, out)
    finally:
        raw_path.unlink(missing_ok=True)
    return f"edge-tts/{voice}"


def _tts_piper(text: str, language: str, speed: float, reference, out: Path) -> str:
    _require("piper", "piper-tts")

    def build() -> dict:
        from huggingface_hub import hf_hub_download, list_repo_files
        from piper.voice import PiperVoice

        repo = "rhasspy/piper-voices"
        candidates = [
            name for name in list_repo_files(repo)
            if name.startswith(f"{language}/") and name.endswith(".onnx")
        ]
        if not candidates:
            raise HTTPException(
                status_code=400,
                detail=f"Piper has no voice for '{language}'. Use 'mms' or 'edge' instead.",
            )
        # Prefer the medium build: low is noticeably rougher, high is much slower.
        candidates.sort(key=lambda name: (0 if "/medium/" in name else 1, name))
        chosen = candidates[0]
        onnx = hf_hub_download(repo, chosen)
        config = hf_hub_download(repo, f"{chosen}.json")
        return {"voice": PiperVoice.load(onnx, config_path=config), "name": chosen}

    slot = _tts_slot(f"piper:{language}", build)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as raw:
        raw_path = Path(raw.name)
    try:
        import wave

        with wave.open(str(raw_path), "wb") as handle:
            slot["voice"].synthesize(text, handle)
        _ffmpeg(raw_path, out, speed)
    finally:
        raw_path.unlink(missing_ok=True)
    return f"piper/{slot['name']}"


def _tts_xtts(text: str, language: str, speed: float, reference, out: Path) -> str:
    _require("TTS", "coqui-tts")
    sample = _needs_reference(reference, "xtts_v2")
    if language not in _XTTS_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"XTTS v2 does not speak '{language}' (Vietnamese included). "
                f"Use 'vixtts' for Vietnamese or 'mms'/'edge' otherwise."
            ),
        )

    def build() -> dict:
        from TTS.api import TTS

        return {"tts": TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(DEVICE)}

    slot = _tts_slot("xtts_v2", build)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as raw:
        raw_path = Path(raw.name)
    try:
        slot["tts"].tts_to_file(
            text=text,
            speaker_wav=str(sample),
            language="zh-cn" if language == "zh" else language,
            file_path=str(raw_path),
        )
        _ffmpeg(raw_path, out, speed)
    finally:
        raw_path.unlink(missing_ok=True)
    return "coqui/xtts_v2"


def _tts_vixtts(text: str, language: str, speed: float, reference, out: Path) -> str:
    _require("TTS", "coqui-tts")
    sample = _needs_reference(reference, "vixtts")
    if language != "vi":
        raise HTTPException(
            status_code=400,
            detail=f"viXTTS is a Vietnamese checkpoint and cannot speak '{language}'.",
        )

    def build() -> dict:
        from huggingface_hub import snapshot_download
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

        folder = snapshot_download("capleaf/viXTTS")
        config = XttsConfig()
        config.load_json(str(Path(folder) / "config.json"))
        model = Xtts.init_from_config(config)
        model.load_checkpoint(config, checkpoint_dir=folder, use_deepspeed=False)
        return {"model": model.to(DEVICE), "config": config}

    slot = _tts_slot("vixtts", build)
    with torch.inference_mode():
        result = slot["model"].synthesize(
            text, slot["config"], speaker_wav=str(sample), language="vi"
        )
    _write_wave(result["wav"], 24000, out, speed)
    return "capleaf/viXTTS"


def _f5_synthesize(checkpoint: str, text: str, sample: Path, out: Path, speed: float) -> str:
    _require("f5_tts", "f5-tts")

    def build() -> dict:
        from f5_tts.api import F5TTS

        return {"f5": F5TTS(model=checkpoint)}

    slot = _tts_slot(f"f5:{checkpoint}", build)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as raw:
        raw_path = Path(raw.name)
    try:
        # An empty ref_text makes F5 transcribe the reference itself.
        slot["f5"].infer(
            ref_file=str(sample), ref_text="", gen_text=text, file_wave=str(raw_path)
        )
        _ffmpeg(raw_path, out, speed)
    finally:
        raw_path.unlink(missing_ok=True)
    return f"f5-tts/{checkpoint}"


def _tts_f5_base(text: str, language: str, speed: float, reference, out: Path) -> str:
    sample = _needs_reference(reference, "f5_base")
    if language not in {"en", "zh"}:
        raise HTTPException(
            status_code=400,
            detail=f"F5-TTS base speaks English and Chinese only, not '{language}'.",
        )
    return _f5_synthesize("F5TTS_v1_Base", text, sample, out, speed)


def _tts_f5_vi(text: str, language: str, speed: float, reference, out: Path) -> str:
    sample = _needs_reference(reference, "f5_vi")
    if language != "vi":
        raise HTTPException(
            status_code=400,
            detail=f"The F5 Vietnamese checkpoint cannot speak '{language}'.",
        )
    return _f5_synthesize(
        os.getenv("F5_VI_MODEL", "hynt/F5-TTS-Vietnamese-ViVoice"), text, sample, out, speed
    )


TTS_ENGINES = {
    "mms":     {"run": _tts_mms,      "package": None,        "reference": False},
    "edge":    {"run": _tts_edge,     "package": "edge-tts",  "reference": False},
    "piper":   {"run": _tts_piper,    "package": "piper-tts", "reference": False},
    "xtts_v2": {"run": _tts_xtts,     "package": "coqui-tts", "reference": True},
    "vixtts":  {"run": _tts_vixtts,   "package": "coqui-tts", "reference": True},
    "f5_vi":   {"run": _tts_f5_vi,    "package": "f5-tts",    "reference": True},
    "f5_base": {"run": _tts_f5_base,  "package": "f5-tts",    "reference": True},
}


def _store_reference(upload: UploadFile) -> str:
    reference_id = uuid.uuid4().hex[:16]
    target = _REFERENCE_DIR / f"{reference_id}.wav"
    with tempfile.NamedTemporaryFile(
        suffix=Path(upload.filename or "ref.wav").suffix or ".wav", delete=False
    ) as raw:
        shutil.copyfileobj(upload.file, raw)
        raw_path = Path(raw.name)
    try:
        # Cloning models expect a clean mono clip; normalise whatever arrives.
        _ffmpeg(raw_path, target)
    finally:
        raw_path.unlink(missing_ok=True)

    stored = sorted(_REFERENCE_DIR.glob("*.wav"), key=lambda p: p.stat().st_mtime)
    for stale in stored[:-_REFERENCE_LIMIT]:
        stale.unlink(missing_ok=True)
    return reference_id


def _reference_path(reference_id: str | None) -> Path | None:
    if not reference_id:
        return None
    candidate = _REFERENCE_DIR / f"{Path(reference_id).name}.wav"
    return candidate if candidate.exists() else None


@app.get("/health")
def health() -> dict:
    return {
        "status": "ready",
        "device": DEVICE,
        "services": ["transcription", "translation", "speech"],
        "whisper_model": _whisper_name or WHISPER_MODEL,
        "whisper_models": sorted(WHISPER_MODELS),
        "translation_model": NLLB_MODEL,
        "translation_engines": sorted(TRANSLATION_ENGINES),
        "tts_engines": {
            name: {
                "available": spec["package"] is None
                or importlib.util.find_spec(spec["package"].replace("-", "_")) is not None,
                "package": spec["package"],
                "needs_reference": spec["reference"],
            }
            for name, spec in TTS_ENGINES.items()
        },
        "languages": [
            {"code": code, "name": values[0]} for code, values in LANGUAGES.items()
        ],
    }


@app.post("/reference")
def upload_reference(
    audio: UploadFile = File(...),
    authorization: str | None = Header(None),
) -> dict:
    _authorize(authorization)
    return {"reference_id": _store_reference(audio)}


@app.post("/transcribe")
def transcribe(
    audio: UploadFile = File(...),
    language: str = Form("auto"),
    model: str = Form(""),
    authorization: str | None = Header(None),
) -> dict:
    _authorize(authorization)
    requested = language.strip().lower()
    source = None if requested in {"", "auto"} else _validate_language(requested)
    checkpoint = model.strip().lower() or WHISPER_MODEL
    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
            shutil.copyfileobj(audio.file, temporary)
            temporary_path = temporary.name
        with _lock:
            raw_segments, info = _load_whisper(checkpoint).transcribe(
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
        return {
            "source_language": detected,
            "segments": segments,
            "whisper_model": checkpoint,
        }
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
    engine = (request.engine or "nllb").strip().lower()
    if engine not in TRANSLATION_ENGINES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported translation engine '{engine}' "
                   f"(pick one of {sorted(TRANSLATION_ENGINES)})",
        )
    if not request.texts or len(request.texts) > 500:
        raise HTTPException(status_code=400, detail="texts must contain 1 to 500 items")
    if source == target:
        return {"translations": request.texts, "translation_engine": engine}

    with _lock:
        translations = TRANSLATION_ENGINES[engine](request.texts, source, target)
    if len(translations) != len(request.texts):
        raise HTTPException(
            status_code=500, detail="The translation model returned the wrong number of rows"
        )
    return {
        "translations": [text.strip() for text in translations],
        "translation_engine": engine,
    }


@app.post("/synthesize")
def synthesize(
    text: str = Form(...),
    language: str = Form("vi"),
    speed: float = Form(1.0),
    engine: str = Form("mms"),
    reference_id: str = Form(""),
    authorization: str | None = Header(None),
) -> Response:
    _authorize(authorization)
    language = _validate_language(language)
    chosen = (engine or "mms").strip().lower()
    if chosen not in TTS_ENGINES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported voice engine '{chosen}' (pick one of {sorted(TTS_ENGINES)})",
        )
    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    if len(text) > 2000:
        raise HTTPException(status_code=400, detail="Text must be at most 2000 characters")
    if not 0.5 <= speed <= 2.0:
        raise HTTPException(status_code=400, detail="Speed must be between 0.5 and 2.0")

    reference = _reference_path(reference_id)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        output = Path(handle.name)
    try:
        with _lock:
            model_name = TTS_ENGINES[chosen]["run"](text, language, speed, reference, output)
        payload = output.read_bytes()
        if not payload:
            raise HTTPException(status_code=500, detail=f"The '{chosen}' engine produced no audio")
        with sf.SoundFile(io.BytesIO(payload)) as probe:
            sample_rate = int(probe.samplerate)
    finally:
        output.unlink(missing_ok=True)

    return Response(
        content=payload,
        media_type="audio/wav",
        headers={
            "X-TTS-Model": model_name,
            "X-TTS-Engine": chosen,
            "X-TTS-Sample-Rate": str(sample_rate),
        },
    )


# ==========================================================================
# Self-contained pipeline
#
# Runs every stage here, so a caller only needs this URL: no local Docker, no
# n8n, and nothing accumulating on the caller's disk. The trade-off is that
# Colab storage is ephemeral - a dropped session loses queued and running jobs,
# so finished videos should be downloaded promptly.
# ==========================================================================
def _default_jobs_root() -> Path:
    """Colab keeps user files under /content; fall back anywhere else."""
    candidate = Path(os.getenv("JOBS_ROOT", "/content/dubflow-jobs"))
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "dubflow-jobs"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


JOBS_ROOT = _default_jobs_root()
#: Colab disk is finite and shared; keep only the most recent jobs.
JOBS_LIMIT = int(os.getenv("JOBS_LIMIT", "20"))
ALLOWED_VIDEO = {".mp4", ".mkv", ".mov", ".webm", ".m4v"}

_pipeline_queue: queue.Queue = queue.Queue()
_worker: threading.Thread | None = None
_worker_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _job_dir(job_id: str) -> Path:
    if not job_id or not job_id.isalnum() or len(job_id) != 12:
        raise HTTPException(status_code=400, detail="Invalid job_id")
    return JOBS_ROOT / job_id


def _read_job(job_id: str) -> dict:
    path = _job_dir(job_id) / "job.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Unknown job '{job_id}'")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_job(job: dict) -> None:
    path = _job_dir(job["job_id"]) / "job.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _prune_jobs() -> None:
    folders = [p for p in JOBS_ROOT.iterdir() if p.is_dir()]
    folders.sort(key=lambda p: p.stat().st_mtime)
    for stale in folders[:-JOBS_LIMIT]:
        shutil.rmtree(stale, ignore_errors=True)


def _run(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout or "command failed")[-2000:])


def _media_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError((result.stderr or "ffprobe failed")[-500:])
    return float(result.stdout.strip())


def _srt_timestamp(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _write_subtitles(path: Path, segments: list[dict]) -> None:
    blocks = []
    for index, segment in enumerate(segments, start=1):
        blocks.extend([
            str(index),
            f"{_srt_timestamp(segment['start'])} --> {_srt_timestamp(segment['end'])}",
            segment.get("translated_text") or segment["source_text"],
            "",
        ])
    path.write_text("\n".join(blocks), encoding="utf-8")


def _stage_extract(job: dict, folder: Path) -> None:
    video = folder / job["files"]["input"]
    original = folder / "original.wav"
    asr_audio = folder / "asr.wav"
    _run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(video),
        "-vn", "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", str(original),
    ])
    _run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(video),
        "-vn", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(asr_audio),
    ])
    job["duration_seconds"] = round(_media_duration(video), 3)
    job["files"].update({"original_audio": original.name, "asr_audio": asr_audio.name})


def _stage_transcribe(job: dict, folder: Path) -> None:
    requested = job.get("source_language")
    source = None if requested in {None, "", "auto"} else _validate_language(requested)
    with _lock:
        raw_segments, info = _load_whisper(job["whisper_model"]).transcribe(
            str(folder / job["files"]["asr_audio"]),
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
    if not segments:
        raise RuntimeError("Whisper found no speech in the video")
    detected = source or str(info.language)
    if detected not in LANGUAGES:
        raise RuntimeError(f"Detected language '{detected}' is not supported")
    job["source_language"] = detected
    job["segments"] = segments
    (folder / "transcript.json").write_text(
        json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _stage_translate(job: dict, folder: Path) -> None:
    source, target = job["source_language"], job["target_language"]
    texts = [segment["source_text"] for segment in job["segments"]]
    if source == target:
        translations = texts
    else:
        engine = job["translation_engine"]
        with _lock:
            translations = TRANSLATION_ENGINES[engine](texts, source, target)
        if len(translations) != len(texts):
            raise RuntimeError("The translation model returned the wrong number of rows")
    for segment, translated in zip(job["segments"], translations, strict=True):
        segment["translated_text"] = translated.strip()
    subtitle = folder / "translated.srt"
    _write_subtitles(subtitle, job["segments"])
    job["files"]["subtitle"] = subtitle.name


def _stage_synthesize(job: dict, folder: Path) -> None:
    engine = job["tts_engine"]
    target = job["target_language"]
    output_dir = folder / "tts"
    output_dir.mkdir(exist_ok=True)

    reference = None
    if TTS_ENGINES[engine]["reference"]:
        # Same clip the HTTP path builds, but kept in-process: no upload, no
        # reference_id, and it cannot expire mid-job.
        segments = job["segments"]
        start = float(segments[0]["start"])
        span = float(segments[-1]["end"]) - start
        if span < 1.0:
            raise RuntimeError(
                "The video holds less than a second of speech, too little to clone a voice"
            )
        reference = folder / "reference.wav"
        _run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{start:.3f}", "-t", f"{min(12.0, span):.3f}",
            "-i", str(folder / job["files"]["original_audio"]),
            "-vn", "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", str(reference),
        ])
        job["files"]["reference"] = reference.name

    models: dict[str, int] = {}
    for segment in job["segments"]:
        text = (segment.get("translated_text") or "").strip()
        if not text:
            continue
        output = output_dir / f"{segment['id']:04d}.wav"
        with _lock:
            model_name = TTS_ENGINES[engine]["run"](text, target, 1.0, reference, output)
        segment["tts_file"] = str(output.relative_to(folder))
        segment["tts_duration"] = round(_media_duration(output), 3)
        segment["tts_model"] = model_name
        models[model_name] = models.get(model_name, 0) + 1
    job["models_used"] = models


def _stage_mix(job: dict, folder: Path) -> None:
    voiced = [segment for segment in job["segments"] if segment.get("tts_file")]
    if not voiced:
        raise RuntimeError("No generated speech is available")

    dubbed = folder / "dubbed.wav"
    command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    for segment in voiced:
        command.extend(["-i", str(folder / segment["tts_file"])])

    filters, labels = [], []
    for index, segment in enumerate(voiced):
        delay = max(0, round(float(segment["start"]) * 1000))
        filters.append(
            f"[{index}:a]aresample=48000,asetpts=PTS-STARTPTS,"
            f"adelay={delay}|{delay}[s{index}]"
        )
        labels.append(f"[s{index}]")
    duration = float(job["duration_seconds"])
    filters.append(
        f"{''.join(labels)}amix=inputs={len(labels)}:duration=longest:normalize=0,"
        f"apad=whole_dur={duration:.3f},atrim=0:{duration:.3f}[dub]"
    )
    command.extend([
        "-filter_complex", ";".join(filters), "-map", "[dub]",
        "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", str(dubbed),
    ])
    _run(command)

    # No Demucs here either: the original track stays quiet underneath, which
    # is a voice-over rather than a studio dub.
    final_audio = folder / "final.wav"
    _run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(folder / job["files"]["original_audio"]), "-i", str(dubbed),
        "-filter_complex",
        "[0:a]volume=0.25[background];[1:a]volume=1.5[voice];"
        "[background][voice]amix=inputs=2:duration=first:normalize=0,"
        "loudnorm=I=-16:TP=-1.5:LRA=11[out]",
        "-map", "[out]", "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le",
        str(final_audio),
    ])
    job["files"].update({"dubbed_audio": dubbed.name, "final_audio": final_audio.name})


def _stage_render(job: dict, folder: Path) -> None:
    output = folder / f"dubbed_{job['target_language']}.mp4"
    duration = float(job["duration_seconds"])
    base = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(folder / job["files"]["input"]),
        "-i", str(folder / job["files"]["final_audio"]),
        "-i", str(folder / job["files"]["subtitle"]),
        "-map", "0:v:0", "-map", "1:a:0", "-map", "2:0",
    ]
    finish = [
        "-c:a", "aac", "-b:a", "192k", "-c:s", "mov_text",
        "-metadata:s:s:0", f"language={job['target_language']}",
        "-t", f"{duration:.3f}", "-movflags", "+faststart", str(output),
    ]
    try:
        _run(base + ["-c:v", "copy"] + finish)
    except RuntimeError:
        # Stream copy refuses containers the muxer cannot take verbatim.
        _run(base + ["-c:v", "libx264", "-preset", "fast", "-crf", "22"] + finish)
    job["files"]["output"] = output.name


PIPELINE = [
    ("extract", _stage_extract),
    ("transcribe", _stage_transcribe),
    ("translate", _stage_translate),
    ("synthesize", _stage_synthesize),
    ("mix", _stage_mix),
    ("render", _stage_render),
]


def _process(job_id: str) -> None:
    """Run every stage for one job, recording progress as it goes."""
    try:
        job = _read_job(job_id)
    except HTTPException:
        return  # Deleted while it sat in the queue.
    folder = _job_dir(job_id)
    job["status"] = "running"
    job["started_at"] = _now()
    _write_job(job)

    for name, run_stage in PIPELINE:
        job["current_stage"] = name
        _write_job(job)
        try:
            run_stage(job, folder)
        except Exception as exc:  # noqa: BLE001 - recorded, then reported by the API
            job["status"] = "failed"
            job["current_stage"] = None
            job["error"] = {"stage": name, "message": str(exc)[:2000]}
            job["finished_at"] = _now()
            _write_job(job)
            return
        job.setdefault("completed_stages", []).append(name)
        _write_job(job)

    job["status"] = "completed"
    job["current_stage"] = None
    job.pop("error", None)
    job["finished_at"] = _now()
    _write_job(job)


def _worker_loop() -> None:
    while True:
        job_id = _pipeline_queue.get()
        try:
            _process(job_id)
        finally:
            _pipeline_queue.task_done()


def _ensure_worker() -> None:
    """One worker, so jobs queue instead of fighting over a single GPU."""
    global _worker
    with _worker_lock:
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_worker_loop, daemon=True)
            _worker.start()


def _public_job(job: dict) -> dict:
    total = len(PIPELINE)
    done = len(job.get("completed_stages", []))
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "current_stage": job.get("current_stage"),
        "completed_stages": job.get("completed_stages", []),
        "progress": f"{done}/{total}",
        "source_language": job.get("source_language"),
        "target_language": job.get("target_language"),
        "whisper_model": job.get("whisper_model"),
        "translation_engine": job.get("translation_engine"),
        "tts_engine": job.get("tts_engine"),
        "models_used": job.get("models_used"),
        "duration_seconds": job.get("duration_seconds"),
        "segments": len(job.get("segments", [])),
        "created_at": job.get("created_at"),
        "finished_at": job.get("finished_at"),
        "error": job.get("error"),
        "download_url": f"/jobs/{job['job_id']}/download"
        if job["status"] == "completed" else None,
        "subtitle_url": f"/jobs/{job['job_id']}/subtitle"
        if job.get("files", {}).get("subtitle") else None,
    }


@app.post("/jobs", status_code=202)
def create_job(
    video: UploadFile = File(...),
    target_language: str = Form("vi"),
    source_language: str = Form("auto"),
    model: str = Form(""),
    translation_engine: str = Form("nllb"),
    tts_engine: str = Form("mms"),
    authorization: str | None = Header(None),
) -> dict:
    """Dub a video end to end. Returns immediately; poll GET /jobs/{id}."""
    _authorize(authorization)
    target = _validate_language(target_language)
    source = (source_language or "auto").strip().lower()
    if source not in {"", "auto"}:
        source = _validate_language(source)
    checkpoint = (model or WHISPER_MODEL).strip().lower()
    if checkpoint not in WHISPER_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported Whisper model '{checkpoint}' "
                   f"(pick one of {sorted(WHISPER_MODELS)})",
        )
    mt_engine = (translation_engine or "nllb").strip().lower()
    if mt_engine not in TRANSLATION_ENGINES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported translation engine '{mt_engine}' "
                   f"(pick one of {sorted(TRANSLATION_ENGINES)})",
        )
    voice = (tts_engine or "mms").strip().lower()
    if voice not in TTS_ENGINES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported voice engine '{voice}' (pick one of {sorted(TTS_ENGINES)})",
        )
    suffix = Path(video.filename or "input.mp4").suffix.lower() or ".mp4"
    if suffix not in ALLOWED_VIDEO:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video container '{suffix}' "
                   f"(use one of {sorted(ALLOWED_VIDEO)})",
        )

    job_id = uuid.uuid4().hex[:12]
    folder = JOBS_ROOT / job_id
    folder.mkdir(parents=True)
    source_path = folder / f"input{suffix}"
    with source_path.open("wb") as handle:
        shutil.copyfileobj(video.file, handle, length=8 * 1024 * 1024)

    job = {
        "job_id": job_id,
        "status": "queued",
        "current_stage": None,
        "completed_stages": [],
        "created_at": _now(),
        "source_filename": video.filename,
        "source_language": None if source in {"", "auto"} else source,
        "target_language": target,
        "whisper_model": checkpoint,
        "translation_engine": mt_engine,
        "tts_engine": voice,
        "files": {"input": source_path.name},
        "segments": [],
    }
    _write_job(job)
    _prune_jobs()
    _ensure_worker()
    _pipeline_queue.put(job_id)
    return {**_public_job(job), "queued_ahead": _pipeline_queue.qsize() - 1}


@app.get("/jobs")
def list_jobs(authorization: str | None = Header(None)) -> dict:
    _authorize(authorization)
    jobs = []
    for folder in sorted(
        (p for p in JOBS_ROOT.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ):
        manifest = folder / "job.json"
        if manifest.exists():
            jobs.append(_public_job(json.loads(manifest.read_text(encoding="utf-8"))))
    return {"jobs": jobs, "queued": _pipeline_queue.qsize(), "limit": JOBS_LIMIT}


@app.get("/jobs/{job_id}")
def get_job(job_id: str, authorization: str | None = Header(None)) -> dict:
    _authorize(authorization)
    return _public_job(_read_job(job_id))


@app.get("/jobs/{job_id}/download")
def download_job(job_id: str, authorization: str | None = Header(None)):  # noqa: ANN201
    _authorize(authorization)
    job = _read_job(job_id)
    name = job.get("files", {}).get("output")
    if not name:
        raise HTTPException(
            status_code=409,
            detail=f"Job '{job_id}' is {job['status']}, the video is not ready",
        )
    return FileResponse(_job_dir(job_id) / name, media_type="video/mp4", filename=name)


@app.get("/jobs/{job_id}/subtitle")
def download_subtitle(job_id: str, authorization: str | None = Header(None)):  # noqa: ANN201
    _authorize(authorization)
    job = _read_job(job_id)
    name = job.get("files", {}).get("subtitle")
    if not name:
        raise HTTPException(status_code=409, detail="Subtitles are not ready")
    return FileResponse(
        _job_dir(job_id) / name, media_type="application/x-subrip", filename=name
    )


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str, authorization: str | None = Header(None)) -> dict:
    _authorize(authorization)
    folder = _job_dir(job_id)
    if not folder.exists():
        raise HTTPException(status_code=404, detail=f"Unknown job '{job_id}'")
    shutil.rmtree(folder, ignore_errors=True)
    return {"job_id": job_id, "deleted": True}
