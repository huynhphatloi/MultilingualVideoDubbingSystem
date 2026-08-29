"""Structured logging with per-request/per-job correlation ids."""
from __future__ import annotations

import contextvars
import json
import logging
import sys
import time
import uuid
from typing import Any

from app.core.config import settings

request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")
job_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("job_id", default="-")
stage_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("stage", default="-")

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "asctime", "message", "taskName",
}


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get()
        record.job_id = job_id_ctx.get()
        record.stage = stage_ctx.get()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
            "job_id": getattr(record, "job_id", "-"),
            "stage": getattr(record, "stage", "-"),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and key not in payload:
                try:
                    json.dumps(value)
                    payload[key] = value
                except (TypeError, ValueError):
                    payload[key] = repr(value)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = (
            f"{self.formatTime(record, '%H:%M:%S')} "
            f"{record.levelname:<7} "
            f"[job={getattr(record, 'job_id', '-')} stage={getattr(record, 'stage', '-')}] "
            f"{record.name}: {record.getMessage()}"
        )
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def setup_logging() -> None:
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if settings.log_json else TextFormatter())
    handler.addFilter(ContextFilter())
    root.addHandler(handler)
    root.setLevel(settings.log_level.upper())

    for noisy in ("urllib3", "botocore", "s3transfer", "numba", "matplotlib", "speechbrain"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


class stage_context:  # noqa: N801 - used as a context manager
    """Bind job_id/stage to every log line emitted inside the block."""

    def __init__(self, job_id: str | None = None, stage: str | None = None) -> None:
        self.job_id = job_id
        self.stage = stage
        self._tokens: list[Any] = []

    def __enter__(self) -> stage_context:
        if self.job_id:
            self._tokens.append((job_id_ctx, job_id_ctx.set(self.job_id)))
        if self.stage:
            self._tokens.append((stage_ctx, stage_ctx.set(self.stage)))
        return self

    def __exit__(self, *exc: Any) -> None:
        for var, token in reversed(self._tokens):
            var.reset(token)
