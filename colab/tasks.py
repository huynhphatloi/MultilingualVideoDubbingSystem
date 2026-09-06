"""Background tasks for the stage endpoints.

A Cloudflare quick tunnel gives up on a request that has not answered in about
a hundred seconds, and transcribing or separating a real video takes longer
than that - the caller then sees a 524 that says nothing about the job, which
is still running on the GPU with nobody left to receive it.

So the stage endpoints accept `async_mode`. The work moves to a background
thread, the request returns a task id immediately, and the caller polls. No
single request is ever long enough to be cut off, and the length of the job
stops mattering.

Model loading stays serialised: every task body takes `jobs.lock`, the same
lock the synchronous routes and the end-to-end pipeline take, so a GPU that
holds one model at a time keeps doing exactly that.
"""
from __future__ import annotations

import threading
import traceback
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional

from core.errors import ServiceError

#: Finished tasks are kept so a caller that polls late still gets its answer.
#: Colab sessions are short-lived; this only has to outlive one pipeline.
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
    """Drop the oldest finished tasks. Running ones are never evicted.

    A task that produced a file - a separated stem, say - owns it, so evicting
    the task deletes it. Colab's disk is small and a stem is the largest thing
    this service writes.
    """
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
    """Tie temporary files to a task, so evicting it removes them."""
    with _lock:
        task = _tasks.get(task_id)
        if task is not None:
            task["files"].extend(str(path) for path in paths)


def submit(work: Callable[[], Dict], label: str = "") -> Dict:
    """Run `work` in the background and return the task's public state.

    `work` returns the JSON payload the synchronous route would have returned.
    A ServiceError it raises keeps its status code, so an async caller learns
    the same thing a synchronous one would.
    """
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
            # A stage whose answer is a file hands back its path; the task owns
            # it from here, so evicting the task removes it.
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
    """What the poller sees. The result is inlined once the task is done, so a
    successful poll is the only round trip the caller needs."""
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
    """Drop every task. For the tests."""
    with _lock:
        _tasks.clear()
        _order.clear()


def result_or_raise(task_id: str) -> Optional[Dict]:
    """The payload of a finished task, or the failure re-raised as it arrived."""
    task = get(task_id)
    if task["status"] == FAILED:
        raise ServiceError(task["error"] or "Task failed", status_code=task["status_code"] or 500)
    return task["result"] if task["status"] == DONE else None
