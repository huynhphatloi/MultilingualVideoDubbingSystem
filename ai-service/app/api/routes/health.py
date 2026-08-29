"""Liveness / readiness / introspection."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

from app.core import languages
from app.core.config import settings
from app.core.device import device_report
from app.core.paths import cache_report
from app.services import ffmpeg, remote_gpu
from app.services.models_registry import loaded, unload, unload_all
from app.services.translation import router as translation_router
from app.services.tts import router as tts_router

router = APIRouter(tags=["health"])


@router.get("/health", summary="Liveness probe")
def health() -> dict:
    return {"status": "ok", "service": settings.app_name, "version": settings.app_version}


@router.get("/health/ready", summary="Readiness probe (checks ffmpeg + storage + db)")
def ready() -> dict:
    checks: dict[str, object] = {}

    try:
        ffmpeg.ensure_available()
        checks["ffmpeg"] = "ok"
    except Exception as exc:  # noqa: BLE001
        checks["ffmpeg"] = f"error: {exc}"

    try:
        from app.storage import get_storage

        storage = get_storage()
        storage.list("jobs/")
        checks["storage"] = f"ok ({storage.name})"
    except Exception as exc:  # noqa: BLE001
        checks["storage"] = f"error: {exc}"

    try:
        from sqlalchemy import text

        from app.jobs.db import get_engine

        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as exc:  # noqa: BLE001
        checks["database"] = f"error: {exc}"

    caches = cache_report()
    for var, info in caches.items():
        checks[f"cache:{var}"] = (
            f"ok ({info['path']})" if info["writable"] else f"error: not writable ({info['path']})"
        )

    if settings.remote_asr_enabled or settings.remote_translation_enabled:
        # A dead tunnel is not "not ready" - the pipeline still falls back to
        # local models. It IS worth reporting, because that fallback is exactly
        # what re-downloads the gigabytes this setup avoids.
        info = remote_gpu.health()
        checks["remote_gpu"] = ("ok" if info.get("reachable", True) and info.get("configured")
                                else f"warn: {info.get('error', 'not configured')}")

    healthy = all(str(v).startswith("ok") for v in checks.values())
    return {"status": "ready" if healthy else "degraded", "checks": checks}


@router.get("/health/device", summary="Runtime device report")
def device() -> dict:
    return device_report()


@router.get("/health/models", summary="Which heavy models are currently warm")
def models() -> dict:
    return {
        "loaded": loaded(),
        "translation_engines": translation_router.available(),
        "tts_adapters": tts_router.available(),
        # Which stages run off this machine, and whether the tunnel is alive.
        # Worth surfacing: the whole point of the remote setup is that nothing
        # is cached locally, so "is it reachable" decides whether the next job
        # downloads 3.7 GB or not.
        "remote": {
            "endpoint": remote_gpu.endpoint() or None,
            "stages": {
                "asr": settings.remote_asr_enabled,
                "translation": settings.remote_translation_enabled,
                "tts": bool(settings.tts_endpoint),
            },
        },
        "local_model_cache": _cache_size(),
        "configured": {
            "whisper": settings.whisper_model,
            "demucs": settings.demucs_model if settings.demucs_enabled else None,
            "diarization": settings.diarization_model if settings.diarization_enabled else None,
            "translation_primary": settings.translation_primary,
            "translation_fallback": settings.translation_fallback,
            "xtts": settings.xtts_model,
        },
    }


@router.delete("/health/models", summary="Free memory by unloading cached models")
def unload_models(name: str | None = None) -> dict:
    if name:
        return {"unloaded": 1 if unload(name) else 0, "model": name}
    return {"unloaded": unload_all()}


def _cache_size() -> dict:
    """How much disk the model caches are actually using.

    The remote setup exists to keep this at zero, and a number is the only way
    to know whether it is working - a fallback to a local engine downloads
    gigabytes quietly and nothing else would say so.
    """
    from app.core.paths import cache_report

    total = 0
    for info in cache_report().values():
        root = Path(info["path"])
        if root.exists():
            total += sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
    return {"bytes": total, "human": f"{total / 1_073_741_824:.2f} GB"}


@router.get("/languages", tags=["metadata"], summary="Supported language catalog")
def language_catalog() -> dict:
    return {"languages": languages.catalog(), "count": len(languages.catalog())}
