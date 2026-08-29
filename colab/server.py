"""The GPU-side endpoint. One process, every engine, chosen per request.

Serves every stage whose model would otherwise sit on the laptop's disk:

    GET  /health      -> catalogue: which engines exist, what they speak
    GET  /engines     -> the same list, without the server fields
    POST /transcribe  -> raw Whisper windows        (saves 1.4 GB locally)
         form: language?, model, word_timestamps, beam_size, vad_filter
         file: audio                                (Opus, ~1.7 MB per 10 min)
    POST /translate   -> translations + candidates  (saves 2.3 GB locally)
         json: {source_language, target_language, engine, items:[{text, char_budget}]}
    POST /synthesize  -> audio/wav
         form: text, language, speed, engine?, speaker_id?, reference_id?
         file: reference?         (only on the first call for a voice)
         409 {"error": "missing_reference"} -> client re-uploads and retries

3.7 GB of that is Whisper plus NLLB, which is the whole reason those two moved:
they are the largest things the pipeline downloads, and this machine has no
CUDA to run them on anyway.

Two things make this worth a file rather than a notebook cell.

**Engine per request.** ``engine=vixtts`` on one call and ``engine=f5_vi`` on
the next means a full A/B over a real job is two runs with one env var changed,
not two notebook sessions. The engine that actually spoke comes back in
``X-TTS-Model``, so the manifest records it per segment and a mixed run is
still readable afterwards.

**One model resident at a time.** A T4 holds one cloning model comfortably and
three not at all. Switching engines unloads the previous one, so the failure
mode of a comparison run is "the first call after a switch is slow" instead of
an OOM twenty segments in. Set ``ALLOW_MULTI_RESIDENT=1`` on an A100 to keep
them warm.

**Reference caching.** The pipeline calls this once or twice per segment and
every call for one character carries the same multi-second wav. Files are
addressed by sha1 after the first upload; a restarted server answers 409 and
the client re-uploads exactly the references still in use.
"""
from __future__ import annotations

import hashlib
import itertools
import logging
import os
import threading
from pathlib import Path

import engines as registry
import stages
from fastapi import FastAPI, File, Form, Header, HTTPException, Response, UploadFile

log = logging.getLogger("tts-server")

REF_DIR = Path(os.environ.get("REF_DIR", "/content/refs"))
OUT_DIR = Path(os.environ.get("OUT_DIR", "/content/out"))
AUTH_TOKEN = os.environ.get("AUTH_TOKEN", "")
DEFAULT_ENGINE = os.environ.get("DEFAULT_ENGINE", "vixtts")
#: Sample rate advertised to the client. It resamples if an engine disagrees,
#: so this is a hint, not a contract.
SAMPLE_RATE = int(os.environ.get("SAMPLE_RATE", "24000"))
ALLOW_MULTI_RESIDENT = os.environ.get("ALLOW_MULTI_RESIDENT", "") == "1"

REF_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Multilingual dubbing - TTS lab")

#: Serialise engine swaps. Two requests naming different engines must not race
#: to unload each other's weights mid-inference.
_swap_lock = threading.Lock()
_resident: str | None = None
#: Output filenames only need to be unique; itertools.count is atomic.
_seq = itertools.count(1)


def _auth(authorization: str | None) -> None:
    if AUTH_TOKEN and authorization != f"Bearer {AUTH_TOKEN}":
        raise HTTPException(status_code=401, detail="bad token")


def _acquire(name: str):  # noqa: ANN202
    """Return a loaded engine, evicting the previous one unless told not to."""
    global _resident
    with _swap_lock:
        engine = registry.get(name)
        if not ALLOW_MULTI_RESIDENT and _resident and _resident != name:
            freed = registry.unload_all(keep=name)
            # Whisper (1.5 GB) and NLLB (2.5 GB) also sit in VRAM once used.
            # A T4 has 16 GB and a cloning engine wants 3; holding all three
            # is how a job OOMs at segment twenty on the machine that
            # transcribed it fine ten minutes earlier.
            freed += stages.free_vram()
            if freed:
                log.info("swapped out %s for %s", ", ".join(freed), name)
        engine.ensure_loaded()
        _resident = name
        return engine


@app.get("/health")
def health() -> dict:
    """Catalogue. Deliberately answerable with nothing loaded and no GPU."""
    catalogue = registry.describe_all()
    default = registry.get(DEFAULT_ENGINE)
    return {
        # Flat fields first: this is what the current client reads.
        "engine": DEFAULT_ENGINE,
        "languages": list(default.languages),
        "voice_cloning": default.voice_cloning,
        "sample_rate": SAMPLE_RATE,
        # The catalogue a comparison run needs.
        "engines": catalogue,
        "resident": _resident,
        "stages": {"transcribe": True, "translate": True, "synthesize": True},
        "stage_models_loaded": stages.loaded(),
        "multi_resident": ALLOW_MULTI_RESIDENT,
        "device": _device(),
        "references_cached": len(list(REF_DIR.glob("*.wav"))),
    }


@app.get("/engines")
def list_engines() -> dict:
    return {"engines": registry.describe_all(), "default": DEFAULT_ENGINE}


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    language: str = Form(""),
    model: str = Form("medium"),
    word_timestamps: str = Form("true"),
    beam_size: int = Form(5),
    vad_filter: str = Form("true"),
    condition_on_previous_text: str = Form("false"),
    authorization: str | None = Header(None),
) -> dict:
    """Whisper on the GPU. Returns raw windows - see stages.py on why."""
    _auth(authorization)
    upload = OUT_DIR / f"asr_{next(_seq):06d}{Path(audio.filename or 'a.opus').suffix}"
    upload.write_bytes(await audio.read())
    try:
        return stages.transcribe(
            str(upload), language=language or None, model=model,
            word_timestamps=_flag(word_timestamps), beam_size=beam_size,
            vad_filter=_flag(vad_filter),
            condition_on_previous_text=_flag(condition_on_previous_text),
        )
    except Exception as exc:  # noqa: BLE001 - 5xx is retryable, and should be
        log.exception("transcribe failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        upload.unlink(missing_ok=True)


@app.post("/translate")
def translate(body: dict, authorization: str | None = Header(None)) -> dict:
    """NLLB / SeamlessM4T on the GPU. Text in, text out - nothing else moves."""
    _auth(authorization)
    items = body.get("items") or []
    if not items:
        return {"engine": body.get("engine", "nllb"), "results": []}
    try:
        return stages.translate(
            items,
            source_language=body.get("source_language", ""),
            target_language=body.get("target_language", ""),
            engine=body.get("engine") or "nllb",
            num_beams=int(body.get("num_beams") or 4),
            max_new_tokens=int(body.get("max_new_tokens") or 256),
        )
    except ValueError as exc:
        # An unknown language pair is the caller's mistake and will not fix
        # itself on a retry - say so with a 4xx.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("translate failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _flag(value: str) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on")


@app.post("/synthesize")
async def synthesize(
    text: str = Form(...),
    language: str = Form("vi"),
    speed: float = Form(1.0),
    engine: str = Form(""),
    speaker_id: str = Form(""),
    reference_id: str = Form(""),
    reference: UploadFile | None = File(None),
    authorization: str | None = Header(None),
) -> Response:
    _auth(authorization)

    name = engine or DEFAULT_ENGINE
    try:
        selected = registry.get(name)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not selected.supports(language):
        # A considered "no". The client does not retry 4xx, which is correct:
        # retrying an unsupported language just burns the tunnel.
        raise HTTPException(
            status_code=400,
            detail=f"{name} does not speak '{language}' "
                   f"(it has {list(selected.languages)})",
        )

    ref_path = _resolve_reference(selected, reference_id, reference)

    out = OUT_DIR / f"seg_{next(_seq):06d}.wav"
    try:
        loaded = _acquire(name)
        spoken = loaded.speak(text, language, out,
                              reference_wav=ref_path, speed=speed)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - 5xx is retryable, and should be
        log.exception("%s failed", name)
        raise HTTPException(status_code=500, detail=f"{name}: {exc}") from exc

    data = out.read_bytes()
    out.unlink(missing_ok=True)
    return Response(
        content=data,
        media_type="audio/wav",
        headers={
            # Report the engine that spoke, not the transport. A manifest
            # saying `remote: 32` tells you nothing when the point of the run
            # was to compare viXTTS against F5.
            "X-TTS-Model": spoken.engine,
            "X-TTS-Voice-Cloned": str(spoken.voice_cloned).lower(),
            "X-TTS-Sample-Rate": str(spoken.sample_rate),
            "X-TTS-Rtf": f"{spoken.rtf:.3f}",
            "X-TTS-Compute-Seconds": f"{spoken.compute_seconds:.3f}",
        },
    )


def _resolve_reference(engine, reference_id: str,  # noqa: ANN001
                       upload: UploadFile | None) -> Path | None:
    """Find the reference wav, or ask the client to send it again.

    The 409 is the whole point: a Colab runtime that restarted has an empty
    REF_DIR but the client still holds the hash. Answering 409 instead of 400
    tells it to re-upload rather than fail the segment.
    """
    if upload is not None:
        raw = upload.file.read()
        digest = reference_id or hashlib.sha1(raw).hexdigest()  # noqa: S324
        path = REF_DIR / f"{digest}.wav"
        path.write_bytes(raw)
        return path

    if reference_id:
        path = REF_DIR / f"{reference_id}.wav"
        if path.exists():
            return path
        raise HTTPException(status_code=409, detail="missing_reference")

    if engine.needs_reference():
        raise HTTPException(
            status_code=400,
            detail=f"{engine.name} clones a voice and needs a reference wav",
        )
    return None


@app.post("/unload")
def unload(authorization: str | None = Header(None)) -> dict:
    """Free VRAM without restarting the runtime - useful between benchmarks."""
    global _resident
    _auth(authorization)
    freed = registry.unload_all(keep=None) + stages.free_vram()
    _resident = None
    return {"unloaded": freed}


def _device() -> str:
    try:
        import torch

        return (f"cuda:{torch.cuda.get_device_name(0)}"
                if torch.cuda.is_available() else "cpu")
    except Exception:  # noqa: BLE001
        return "unknown"
