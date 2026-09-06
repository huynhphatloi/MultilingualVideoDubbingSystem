"""Job storage and the single-worker queue."""
from __future__ import annotations

import json
import os
import queue
import shutil
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pipeline
from core.errors import InvalidRequest, ServiceError
from core.runtime import slot

ALLOWED_VIDEO = {".mp4", ".mkv", ".mov", ".webm", ".m4v"}
JOB_ID_LENGTH = 12


def _default_root() -> Path:
    candidate = Path(os.getenv("JOBS_ROOT", "/content/dubflow-jobs"))
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "dubflow-jobs"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


ROOT = _default_root()
LIMIT = int(os.getenv("JOBS_LIMIT", "20"))
KEEP_MODELS = os.getenv("DUBFLOW_KEEP_MODELS", "").strip().lower() in {"1", "true", "yes"}

_queue: "queue.Queue[str]" = queue.Queue()
_worker: Optional[threading.Thread] = None
_worker_lock = threading.Lock()
lock = threading.RLock()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id() -> str:
    return uuid.uuid4().hex[:JOB_ID_LENGTH]


def directory(job_id: str) -> Path:
    if not job_id or not job_id.isalnum() or len(job_id) != JOB_ID_LENGTH:
        raise InvalidRequest("Invalid job_id")
    return ROOT / job_id


def read(job_id: str) -> Dict:
    path = directory(job_id) / "job.json"
    if not path.exists():
        raise ServiceError(f"Unknown job '{job_id}'", status_code=404)
    return json.loads(path.read_text(encoding="utf-8"))


def write(job: Dict) -> None:
    path = directory(job["job_id"]) / "job.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def prune() -> None:
    folders = [entry for entry in ROOT.iterdir() if entry.is_dir()]
    folders.sort(key=lambda entry: entry.stat().st_mtime)
    for stale in folders[:-LIMIT]:
        shutil.rmtree(stale, ignore_errors=True)


def listing() -> List[Dict]:
    jobs = []
    for folder in sorted(
        (entry for entry in ROOT.iterdir() if entry.is_dir()),
        key=lambda entry: entry.stat().st_mtime,
        reverse=True,
    ):
        manifest = folder / "job.json"
        if manifest.exists():
            jobs.append(public(json.loads(manifest.read_text(encoding="utf-8"))))
    return jobs


def public(job: Dict) -> Dict:
    planned = job.get("planned_stages") or pipeline.planned_stages(job)
    done = [name for name in job.get("completed_stages", []) if name in planned]
    config = job.get("config") or {}
    tts = config.get("tts") or {}
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "current_stage": job.get("current_stage"),
        "completed_stages": job.get("completed_stages", []),
        "skipped_stages": job.get("skipped_stages", []),
        "planned_stages": planned,
        "progress": f"{len(done)}/{len(planned)}",
        "config": config,
        "source_language": job.get("source_language") or config.get("source_language"),
        "target_language": config.get("target_language"),
        "whisper_model": (config.get("asr") or {}).get("model"),
        "translation_engine": (config.get("translation") or {}).get("model"),
        "tts_engine": tts.get("model"),
        "models_used": job.get("models_used"),
        "speakers": job.get("speakers"),
        "speaker_summary": job.get("speaker_summary"),
        "speaker_voice_map": job.get("speaker_voice_map"),
        "alignment": job.get("alignment"),
        "mix_mode": job.get("mix_mode"),
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


def create(video_name: str, values: Dict, config_public: Dict, save: Callable[[Path], None]) -> Dict:
    suffix = Path(video_name or "input.mp4").suffix.lower() or ".mp4"
    if suffix not in ALLOWED_VIDEO:
        raise InvalidRequest(
            f"Unsupported video container '{suffix}' (use one of {sorted(ALLOWED_VIDEO)})"
        )
    job_id = new_id()
    folder = ROOT / job_id
    folder.mkdir(parents=True)
    source = folder / f"input{suffix}"
    save(source)

    job = {
        "job_id": job_id,
        "status": "queued",
        "current_stage": None,
        "completed_stages": [],
        "skipped_stages": [],
        "created_at": now(),
        "source_filename": video_name,
        "source_language": config_public.get("source_language")
        if config_public.get("source_language") != "auto" else None,
        "config": config_public,
        "request": dict(values),
        "files": {"input": source.name},
        "segments": [],
    }
    job["planned_stages"] = pipeline.planned_stages(job)
    write(job)
    prune()
    return job


def enqueue(job_id: str) -> int:
    ensure_worker()
    _queue.put(job_id)
    return max(0, _queue.qsize() - 1)


def queued() -> int:
    return _queue.qsize()


def process(job_id: str) -> None:
    try:
        job = read(job_id)
    except ServiceError:
        return  # Deleted while it sat in the queue.
    folder = directory(job_id)
    job["status"] = "running"
    job["started_at"] = now()
    job["planned_stages"] = pipeline.planned_stages(job)
    write(job)

    for name in pipeline.STAGE_NAMES:
        if not pipeline.will_run(name, job):
            job.setdefault("skipped_stages", []).append(name)
            write(job)
            continue
        job["current_stage"] = name
        write(job)
        try:
            with lock:
                pipeline.run_stage(name, job, folder)
        except Exception as exc:  # noqa: BLE001 - recorded, then reported by the API
            job["status"] = "failed"
            job["current_stage"] = None
            job["error"] = {"stage": name, "message": str(exc)[:2000]}
            job["finished_at"] = now()
            write(job)
            return
        _mark_completed(job, name)
        write(job)
        if not KEEP_MODELS:
            for released in pipeline.released_after(name, job):
                slot(released).release()

    job["status"] = "completed"
    job["current_stage"] = None
    job.pop("error", None)
    job["finished_at"] = now()
    write(job)


def _mark_completed(job: Dict, name: str) -> None:
    completed = job.setdefault("completed_stages", [])
    if name not in completed:
        completed.append(name)
    skipped = job.get("skipped_stages") or []
    if name in skipped:
        skipped.remove(name)


def _loop() -> None:
    while True:
        job_id = _queue.get()
        try:
            process(job_id)
        finally:
            _queue.task_done()


def ensure_worker() -> None:
    global _worker
    with _worker_lock:
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_loop, daemon=True)
            _worker.start()


def delete(job_id: str) -> None:
    folder = directory(job_id)
    if not folder.exists():
        raise ServiceError(f"Unknown job '{job_id}'", status_code=404)
    shutil.rmtree(folder, ignore_errors=True)
