"""FastAPI AI service.

n8n orchestrates; this service does the work. Every endpoint is synchronous
from the caller's point of view (n8n waits) but reports progress into Postgres
so the web UI can show a live stage tracker.
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import audio, health, jobs, media, speech, subtitle, translation, video
from app.core.config import settings
from app.core.device import configure_threads, device_report
from app.core.errors import PipelineError
from app.core.logging import new_request_id, request_id_ctx, setup_logging
from app.core.paths import ensure_model_cache
from app.jobs.db import init_db
from app.services import ffmpeg

log = logging.getLogger(__name__)

# Runs at import time, on purpose: huggingface_hub freezes its cache location
# the moment it is imported, so this has to happen before any model library
# gets pulled in by a route module.
_MODEL_CACHE = ensure_model_cache()

DESCRIPTION = """
Modular AI/media backend for the **Multilingual Video Translation & Dubbing**
pipeline. The n8n workflow calls these endpoints in order; large media never
travels through n8n - only `job_id` and object keys do.

| Group | Endpoints |
|---|---|
| 1 - Media preprocessing | `/media/extract-audio`, `/media/separate` |
| 2 - Speech understanding | `/speech/transcribe`, `/speech/diarize`, `/speech/merge` |
| 3 - Translation | `/translation/translate`, `/translation/adapt` |
| 4 - Speech generation | `/speech/synthesize` |
| 5 - Synchronization | `/audio/synchronize` |
| 6 - Mixing & render | `/audio/mix`, `/subtitle/generate`, `/video/render` |
"""


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN201
    setup_logging()
    configure_threads()
    log.info("starting %s v%s", settings.app_name, settings.app_version)
    log.info("device report: %s", device_report())
    log.info("model cache: %s", _MODEL_CACHE)

    try:
        ffmpeg.ensure_available()
    except Exception as exc:  # noqa: BLE001
        log.error("ffmpeg check failed: %s", exc)

    init_db()

    from app.storage import get_storage

    try:
        log.info("storage backend: %s", get_storage().name)
    except Exception as exc:  # noqa: BLE001
        log.error("storage init failed: %s", exc)

    yield
    log.info("shutting down")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):  # noqa: ANN001, ANN201
    rid = request.headers.get("x-request-id") or new_request_id()
    token = request_id_ctx.set(rid)
    started = time.perf_counter()
    try:
        response = await call_next(request)
    finally:
        request_id_ctx.reset(token)
    elapsed = int((time.perf_counter() - started) * 1000)
    response.headers["x-request-id"] = rid
    response.headers["x-process-time-ms"] = str(elapsed)
    if request.url.path not in ("/health", "/health/ready"):
        log.info("%s %s -> %s (%d ms)", request.method, request.url.path,
                 response.status_code, elapsed)
    return response


# ------------------------------------------------------- exception handling --
@app.exception_handler(PipelineError)
async def pipeline_error_handler(request: Request, exc: PipelineError):  # noqa: ANN201
    log.warning("pipeline error %s: %s", exc.code, exc.message)
    return JSONResponse(status_code=exc.http_status, content=exc.to_payload())


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):  # noqa: ANN201
    return JSONResponse(
        status_code=422,
        content={"error": {"code": "validation_error", "message": "Invalid request body",
                           "retryable": False, "details": {"errors": exc.errors()}}},
    )


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception):  # noqa: ANN201
    # Full traceback (and any SQL) goes to the log; the client gets a short
    # first line so the browser never displays query parameters.
    log.exception("unhandled error on %s", request.url.path)
    summary = str(exc).strip().splitlines()[0] if str(exc).strip() else exc.__class__.__name__
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "internal_error",
                "message": summary[:300],
                "retryable": True,
                "details": {"type": exc.__class__.__name__,
                            "hint": "full traceback is in the ai-service log"},
            }
        },
    )


# ------------------------------------------------------------------ routes ---
app.include_router(health.router)
app.include_router(jobs.router)
app.include_router(media.router)
app.include_router(speech.router)
app.include_router(translation.router)
app.include_router(audio.router)
app.include_router(subtitle.router)
app.include_router(video.router)


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health",
    }
