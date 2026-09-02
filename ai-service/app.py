"""Small media API used by the n8n dubbing workflow.

The service deliberately has no database, object storage, frontend framework,
diarization, or model router. It also serves the single-file demo UI. Each job
is one directory containing a JSON manifest and the files produced by the seven
visible n8n stages. All model inference is delegated to Google Colab.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

ROOT = Path(os.getenv("DATA_ROOT", Path(__file__).resolve().parent.parent / "data/jobs"))
ROOT.mkdir(parents=True, exist_ok=True)

COLAB_API_URL = os.getenv("COLAB_API_URL", "").rstrip("/")
COLAB_API_TOKEN = os.getenv("COLAB_API_TOKEN", "")
COLAB_API_TIMEOUT = float(os.getenv("COLAB_API_TIMEOUT", "1800"))
N8N_WEBHOOK_URL = os.getenv(
    "N8N_WEBHOOK_URL", "http://n8n:5678/webhook/dubbing/start"
)

APP_DIR = Path(__file__).resolve().parent
FRONTEND_INDEX = Path(
    os.getenv("FRONTEND_INDEX", APP_DIR / "frontend" / "index.html")
)
if not FRONTEND_INDEX.exists():
    # Local development keeps frontend/ beside ai-service/ at repository root.
    FRONTEND_INDEX = APP_DIR.parent / "frontend" / "index.html"

# ISO-639-1 stays the public application code. Colab owns all model-specific
# language mappings.
LANGUAGES = {
    "en": "English", "vi": "Vietnamese", "ja": "Japanese",
    "ko": "Korean", "zh": "Chinese", "fr": "French", "de": "German",
    "es": "Spanish", "pt": "Portuguese", "it": "Italian", "ru": "Russian",
    "nl": "Dutch", "pl": "Polish", "tr": "Turkish", "ar": "Arabic",
    "hi": "Hindi", "id": "Indonesian", "th": "Thai", "cs": "Czech",
    "hu": "Hungarian", "uk": "Ukrainian", "ro": "Romanian",
    "sv": "Swedish", "da": "Danish", "fi": "Finnish", "no": "Norwegian",
    "el": "Greek", "he": "Hebrew", "ms": "Malay", "fa": "Persian",
    "bn": "Bengali", "ta": "Tamil", "te": "Telugu", "ur": "Urdu",
    "sw": "Swahili", "tl": "Tagalog", "km": "Khmer", "lo": "Lao",
    "my": "Burmese", "bg": "Bulgarian", "hr": "Croatian", "sr": "Serbian",
    "sk": "Slovak", "sl": "Slovenian", "ca": "Catalan", "hy": "Armenian",
    "ka": "Georgian", "ne": "Nepali", "si": "Sinhala", "mn": "Mongolian",
}

_JOB_ID = re.compile(r"^[a-f0-9]{12}$")
_ALLOWED_VIDEO = {".mp4", ".mkv", ".mov", ".webm", ".m4v"}
app = FastAPI(title="Simple Multilingual Dubbing API", version="2.0")


class JobRequest(BaseModel):
    job_id: str


def _job_dir(job_id: str) -> Path:
    if not _JOB_ID.fullmatch(job_id):
        raise HTTPException(400, "Invalid job_id")
    path = ROOT / job_id
    if not path.exists():
        raise HTTPException(404, f"Job '{job_id}' does not exist")
    return path


def _manifest_path(job_id: str) -> Path:
    return _job_dir(job_id) / "job.json"


def _load_job(job_id: str) -> dict:
    return json.loads(_manifest_path(job_id).read_text(encoding="utf-8"))


def _save_job(job: dict) -> None:
    path = _job_dir(job["job_id"]) / "job.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


@contextmanager
def _stage(job_id: str, name: str) -> Iterator[tuple[dict, Path]]:
    job = _load_job(job_id)
    folder = _job_dir(job_id)
    job["status"] = "running"
    job["current_stage"] = name
    _save_job(job)
    try:
        yield job, folder
    except Exception as exc:
        message = exc.detail if isinstance(exc, HTTPException) else str(exc)
        job["status"] = "failed"
        job["error"] = {"stage": name, "message": str(message)[:1000]}
        _save_job(job)
        raise
    else:
        job.setdefault("completed_stages", []).append(name)
        job["current_stage"] = None
        job["status"] = "completed" if name == "render" else "running"
        job.pop("error", None)
        _save_job(job)


def _run(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        detail = (result.stderr or result.stdout or "command failed")[-3000:]
        raise RuntimeError(detail)


def _duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError((result.stderr or "ffprobe failed")[-1000:])
    return float(result.stdout.strip())


def _colab_request(path: str, **kwargs):  # noqa: ANN003, ANN202
    """Call the mandatory Colab AI service without any local fallback."""
    if not COLAB_API_URL:
        raise HTTPException(
            503,
            "Google Colab AI service is not configured. Set COLAB_API_URL and "
            "COLAB_API_TOKEN in .env, then restart the stack.",
        )
    if not COLAB_API_URL.startswith(("http://", "https://")):
        raise HTTPException(
            503,
            "COLAB_API_URL must start with http:// or https://. Check that "
            "COLAB_API_URL and COLAB_API_TOKEN were not swapped in .env.",
        )

    import httpx

    headers = kwargs.pop("headers", {})
    if COLAB_API_TOKEN:
        headers["Authorization"] = f"Bearer {COLAB_API_TOKEN}"
    try:
        response = httpx.post(
            f"{COLAB_API_URL}{path}",
            headers=headers,
            timeout=COLAB_API_TIMEOUT,
            **kwargs,
        )
    except httpx.RequestError as exc:
        raise HTTPException(
            503,
            f"Google Colab AI service is unreachable: {str(exc)[:500]}",
        ) from exc
    if response.status_code >= 400:
        raise HTTPException(
            502,
            f"Google Colab AI service returned {response.status_code}: "
            f"{response.text[:500]}",
        )
    return response


def _srt_timestamp(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _write_subtitles(path: Path, segments: list[dict]) -> None:
    blocks = []
    for index, segment in enumerate(segments, start=1):
        blocks.extend(
            [
                str(index),
                f"{_srt_timestamp(segment['start'])} --> "
                f"{_srt_timestamp(segment['end'])}",
                segment.get("translated_text") or segment["source_text"],
                "",
            ]
        )
    path.write_text("\n".join(blocks), encoding="utf-8")


@app.get("/", include_in_schema=False)
def demo_frontend():  # noqa: ANN201
    if not FRONTEND_INDEX.exists():
        raise HTTPException(404, "Demo frontend is not installed")
    return FileResponse(FRONTEND_INDEX, media_type="text/html")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ready",
        "version": "2.0",
        "jobs_root": str(ROOT),
        "languages": len(LANGUAGES),
        "ai_backend": "google-colab",
        "colab_configured": bool(COLAB_API_URL),
        "local_models": False,
    }


@app.get("/languages")
def languages() -> dict:
    return {
        "languages": [
            {"code": code, "name": name}
            for code, name in LANGUAGES.items()
        ]
    }


@app.post("/jobs/upload")
def upload_video(
    file: UploadFile = File(...),
    target_language: str = Form("vi"),
    source_language: str = Form("auto"),
) -> dict:
    target = target_language.strip().lower()
    source = source_language.strip().lower()
    source = None if source in {"", "auto"} else source
    if target not in LANGUAGES:
        raise HTTPException(400, f"Unsupported target language '{target}'")
    if source and source not in LANGUAGES:
        raise HTTPException(400, f"Unsupported source language '{source}'")
    suffix = Path(file.filename or "input.mp4").suffix.lower() or ".mp4"
    if suffix not in _ALLOWED_VIDEO:
        raise HTTPException(400, f"Unsupported video container '{suffix}'")

    job_id = uuid.uuid4().hex[:12]
    folder = ROOT / job_id
    folder.mkdir(parents=True)
    source_path = folder / f"input{suffix}"
    with source_path.open("wb") as output:
        shutil.copyfileobj(file.file, output, length=8 * 1024 * 1024)

    job = {
        "job_id": job_id,
        "status": "running",
        "current_stage": None,
        "completed_stages": ["upload"],
        "source_filename": file.filename,
        "source_language": source,
        "target_language": target,
        "files": {"input": source_path.name},
        "segments": [],
    }
    (folder / "job.json").write_text(
        json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "job_id": job_id,
        "stage": "upload",
        "status": "completed",
        "target_language": target,
    }


@app.post("/stages/extract")
def extract_audio(request: JobRequest) -> dict:
    with _stage(request.job_id, "extract") as (job, folder):
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
        job["duration_seconds"] = round(_duration(video), 3)
        job["files"].update({"original_audio": original.name, "asr_audio": asr_audio.name})
    return {
        "job_id": request.job_id,
        "stage": "extract",
        "status": "completed",
        "duration_seconds": job["duration_seconds"],
    }


@app.post("/stages/transcribe")
def transcribe(request: JobRequest) -> dict:
    with _stage(request.job_id, "transcribe") as (job, folder):
        audio = folder / job["files"]["asr_audio"]
        with audio.open("rb") as stream:
            response = _colab_request(
                "/transcribe",
                files={"audio": (audio.name, stream, "audio/wav")},
                data={
                    # An empty language means auto-detect. The Colab server
                    # forwards anything else straight to Whisper, which rejects
                    # the literal "auto" as an invalid language code.
                    "language": job.get("source_language") or "",
                    "model": "small",
                },
            )
        try:
            payload = response.json()
            if not isinstance(payload, dict):
                raise TypeError("response is not an object")
            segments = payload.get("segments") or payload.get("windows")
            detected = payload.get("source_language") or payload.get("language")
        except (ValueError, KeyError, TypeError) as exc:
            raise RuntimeError("Colab returned an invalid transcription response") from exc
        if not isinstance(segments, list) or not segments:
            raise RuntimeError("Whisper found no speech in the video")
        for index, segment in enumerate(segments):
            segment["id"] = index
        if detected not in LANGUAGES:
            raise RuntimeError(f"Detected language '{detected}' is not configured")
        job["source_language"] = detected
        job["segments"] = segments
        (folder / "transcript.json").write_text(
            json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return {
        "job_id": request.job_id,
        "stage": "transcribe",
        "status": "completed",
        "source_language": job["source_language"],
        "segment_count": len(job["segments"]),
    }


@app.post("/stages/translate")
def translate(request: JobRequest) -> dict:
    with _stage(request.job_id, "translate") as (job, folder):
        source = job["source_language"]
        target = job["target_language"]
        texts = [segment["source_text"] for segment in job["segments"]]

        if source == target:
            translations = texts
        else:
            response = _colab_request(
                "/translate",
                json={
                    "source_language": source,
                    "target_language": target,
                    "texts": texts,
                    "items": [{"text": text} for text in texts],
                },
            )
            try:
                payload = response.json()
                if not isinstance(payload, dict):
                    raise TypeError("response is not an object")
                translations = payload.get("translations")
                if translations is None:
                    translations = [item["text"] for item in payload["results"]]
            except (ValueError, KeyError, TypeError) as exc:
                raise RuntimeError("Colab returned an invalid translation response") from exc
            if len(translations) != len(texts):
                raise RuntimeError("Colab returned the wrong number of translations")

        for segment, translated in zip(job["segments"], translations, strict=True):
            segment["translated_text"] = translated.strip()
        subtitle = folder / "translated.srt"
        _write_subtitles(subtitle, job["segments"])
        job["files"]["subtitle"] = subtitle.name
    return {
        "job_id": request.job_id,
        "stage": "translate",
        "status": "completed",
        "source_language": source,
        "target_language": target,
        "segment_count": len(job["segments"]),
    }


def _colab_speech(
    text: str,
    language: str,
    output: Path,
) -> str:
    # MMS is the non-cloning engine: it speaks every target language and needs
    # no reference sample, unlike the Colab server's cloning default.
    data = {"text": text, "language": language, "speed": "1.0", "engine": "mms"}
    response = _colab_request("/synthesize", data=data)
    output.write_bytes(response.content)
    if not output.exists() or output.stat().st_size == 0:
        raise RuntimeError("Colab TTS returned an empty audio file")
    return response.headers.get("x-tts-model") or "mms"


@app.post("/stages/synthesize")
def synthesize(request: JobRequest) -> dict:
    with _stage(request.job_id, "synthesize") as (job, folder):
        target = job["target_language"]
        output_dir = folder / "tts"
        output_dir.mkdir(exist_ok=True)
        models: dict[str, int] = {}
        for segment in job["segments"]:
            text = (segment.get("translated_text") or "").strip()
            if not text:
                continue
            output = output_dir / f"{segment['id']:04d}.wav"
            model_name = _colab_speech(text, target, output)
            segment["tts_file"] = str(output.relative_to(folder))
            segment["tts_duration"] = round(_duration(output), 3)
            segment["tts_model"] = model_name
            models[model_name] = models.get(model_name, 0) + 1
        job["tts_provider"] = "colab"

    generated = sum(1 for segment in job["segments"] if segment.get("tts_file"))
    return {
        "job_id": request.job_id,
        "stage": "synthesize",
        "status": "completed",
        "generated_segments": generated,
        "provider": job["tts_provider"],
        "models_used": models,
    }


@app.post("/stages/mix")
def mix(request: JobRequest) -> dict:
    with _stage(request.job_id, "mix") as (job, folder):
        voiced = [segment for segment in job["segments"] if segment.get("tts_file")]
        if not voiced:
            raise RuntimeError("No generated speech is available")

        dubbed = folder / "dubbed.wav"
        command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
        for segment in voiced:
            command.extend(["-i", str(folder / segment["tts_file"])])

        filters = []
        labels = []
        for index, segment in enumerate(voiced):
            delay = max(0, round(float(segment["start"]) * 1000))
            label = f"s{index}"
            filters.append(
                f"[{index}:a]aresample=48000,asetpts=PTS-STARTPTS,"
                f"adelay={delay}|{delay}[{label}]"
            )
            labels.append(f"[{label}]")
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

        # The simple pipeline intentionally skips Demucs. The original soundtrack is kept at
        # low volume, producing a simple voice-over rather than a studio dub.
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
    return {
        "job_id": request.job_id,
        "stage": "mix",
        "status": "completed",
        "mode": "voice-over",
        "placed_segments": len(voiced),
    }


@app.post("/stages/render")
def render(request: JobRequest) -> dict:
    with _stage(request.job_id, "render") as (job, folder):
        video = folder / job["files"]["input"]
        audio = folder / job["files"]["final_audio"]
        subtitle = folder / job["files"]["subtitle"]
        output = folder / f"dubbed_{job['target_language']}.mp4"
        duration = float(job["duration_seconds"])
        base = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(video), "-i", str(audio), "-i", str(subtitle),
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
            _run(base + ["-c:v", "libx264", "-preset", "fast", "-crf", "22"] + finish)
        job["files"]["output"] = output.name
        job["output_url"] = f"{os.getenv('PUBLIC_BASE_URL', 'http://localhost:8000')}" \
                            f"/jobs/{request.job_id}/download"
    return {
        "job_id": request.job_id,
        "stage": "render",
        "status": "completed",
        "output_url": job["output_url"],
        "subtitle_url": f"{os.getenv('PUBLIC_BASE_URL', 'http://localhost:8000')}"
                        f"/jobs/{request.job_id}/subtitle",
    }


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    return _load_job(job_id)


@app.post("/jobs/{job_id}/start", status_code=202)
def start_job(job_id: str) -> dict:
    """Ask n8n to orchestrate the six stages after upload."""
    job = _load_job(job_id)
    completed = job.get("completed_stages", [])
    if job.get("status") == "completed" or len(completed) > 1:
        return {
            "job_id": job_id,
            "status": job.get("status", "running"),
            "message": "Job has already started",
        }

    import httpx

    try:
        response = httpx.post(
            N8N_WEBHOOK_URL,
            json={"job_id": job_id},
            timeout=30,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(
            503,
            "n8n did not accept the job. Open http://localhost:5678 and "
            "activate 'Simple Multilingual Dubbing', then try again. "
            f"Technical detail: {str(exc)[:300]}",
        ) from exc

    return {"job_id": job_id, "status": "accepted", "orchestrator": "n8n"}


@app.get("/jobs/{job_id}/download")
def download(job_id: str):  # noqa: ANN201
    job = _load_job(job_id)
    output = _job_dir(job_id) / job.get("files", {}).get("output", "missing")
    if not output.exists():
        raise HTTPException(404, "Rendered video is not available yet")
    return FileResponse(output, media_type="video/mp4", filename=output.name)


@app.get("/jobs/{job_id}/subtitle")
def download_subtitle(job_id: str):  # noqa: ANN201
    job = _load_job(job_id)
    subtitle = _job_dir(job_id) / job.get("files", {}).get("subtitle", "missing")
    if not subtitle.exists():
        raise HTTPException(404, "Subtitle is not available yet")
    return FileResponse(subtitle, media_type="application/x-subrip", filename=subtitle.name)
