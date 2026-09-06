"""Background task storage for long-running model requests."""
from __future__ import annotations

import threading
import traceback
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional

from core.errors import ServiceError

LIMIT = 64

QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
TERMINAL = (DONE, FAILED)

_tasks: "Dict[str, Dict]" = {}
_order: list = []
_lock = threading.Lock()


def new_id() -> str:
    return uuid.uuid4().hex[:16]


def _prune() -> None:
    """Drop the oldest finished tasks and their files."""
    while len(_order) > LIMIT:
        for index, task_id in enumerate(_order):
            if _tasks.get(task_id, {}).get("status") in TERMINAL:
                _order.pop(index)
                stale = _tasks.pop(task_id, None) or {}
                for path in stale.get("files") or []:
                    Path(path).unlink(missing_ok=True)
                break
        else:
            return


def owns(task_id: str, *paths: Path) -> None:
    with _lock:
        task = _tasks.get(task_id)
        if task is not None:
            task["files"].extend(str(path) for path in paths)


def submit(work: Callable[[], Dict], label: str = "") -> Dict:
    """Run work in a background thread and return its task state."""
    task_id = new_id()
    with _lock:
        _tasks[task_id] = {
            "task_id": task_id,
            "label": label,
            "status": QUEUED,
            "result": None,
            "error": None,
            "status_code": None,
            "files": [],
        }
        _order.append(task_id)
        _prune()

    def run() -> None:
        _set(task_id, status=RUNNING)
        try:
            payload = work()
        except ServiceError as exc:
            _set(task_id, status=FAILED, error=exc.message, status_code=exc.status_code)
        except Exception as exc:  # noqa: BLE001 - reported through the task
            traceback.print_exc()
            _set(
                task_id,
                status=FAILED,
                error=f"{exc.__class__.__name__}: {exc}"[:600],
                status_code=500,
            )
        else:
            _set(task_id, status=DONE, result=payload)
            if isinstance(payload, dict) and payload.get("file"):
                owns(task_id, payload["file"])

    threading.Thread(target=run, daemon=True).start()
    return public(task_id)


def _set(task_id: str, **fields: object) -> None:
    with _lock:
        task = _tasks.get(task_id)
        if task is not None:
            task.update(fields)


def get(task_id: str) -> Dict:
    with _lock:
        task = _tasks.get(task_id)
    if task is None:
        raise ServiceError(f"Unknown task '{task_id}'", status_code=404)
    return task


def public(task_id: str) -> Dict:
    task = get(task_id)
    payload = {
        "task_id": task["task_id"],
        "status": task["status"],
        "label": task["label"],
        "done": task["status"] in TERMINAL,
    }
    if task["status"] == DONE:
        payload["result"] = task["result"]
    if task["status"] == FAILED:
        payload["error"] = task["error"]
        payload["status_code"] = task["status_code"]
    return payload


def running() -> int:
    with _lock:
        return sum(1 for task in _tasks.values() if task["status"] in (QUEUED, RUNNING))


def reset() -> None:
    with _lock:
        _tasks.clear()
        _order.clear()


def result_or_raise(task_id: str) -> Optional[Dict]:
    task = get(task_id)
    if task["status"] == FAILED:
        raise ServiceError(task["error"] or "Task failed", status_code=task["status_code"] or 500)
    return task["result"] if task["status"] == DONE else None
