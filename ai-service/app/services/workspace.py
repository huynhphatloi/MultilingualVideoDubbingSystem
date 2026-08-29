"""Per-job scratch directory + pull/push helpers.

Models need real files on disk; storage speaks in object keys. This class is
the bridge, and it keeps every temporary file under one predictable folder so
cleanup is trivial.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

from app.core.config import settings
from app.storage import get_storage
from app.storage.layout import JobLayout

log = logging.getLogger(__name__)


class JobWorkspace:
    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        self.layout = JobLayout(job_id)
        self.storage = get_storage()
        self.root = Path(settings.scratch_root) / job_id
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- paths ---
    def path(self, *parts: str) -> Path:
        p = self.root.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def subdir(self, name: str) -> Path:
        d = self.root / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    # -------------------------------------------------------------- pull ---
    def local_path_for(self, key: str) -> Path:
        """Scratch path that mirrors the object key.

        Deriving the name from the key (rather than letting each caller pick
        one) is what makes the scratch cache actually work: the same object
        always lands on the same file. Callers used to pass their own names and
        the disagreements were expensive - the source video was stored twice per
        job, and `speech.wav` three times under three aliases.
        """
        prefix = f"{self.layout.prefix}/"
        relative = key[len(prefix):] if key.startswith(prefix) else key.replace("/", "_")
        return self.path(*relative.split("/"))

    def pull(self, key: str) -> Path:
        """Fetch an artifact into the scratch dir (cached across calls)."""
        dst = self.local_path_for(key)
        if dst.exists() and dst.stat().st_size > 0:
            return dst
        log.debug("pulling %s -> %s", key, dst)
        return self.storage.get_file(key, dst)

    def pull_json(self, key: str):  # noqa: ANN201
        return self.storage.get_json(key)

    # -------------------------------------------------------------- push ---
    def push(self, local_path: str | Path, key: str, content_type: str | None = None) -> str:
        self.storage.put_file(key, local_path, content_type=content_type)
        return key

    def push_json(self, payload, key: str) -> str:  # noqa: ANN001
        self.storage.put_json(key, payload)
        return key

    # ------------------------------------------------------------ tidy up---
    def cleanup(self) -> None:
        """Delete this job's scratch dir.

        Everything in here is a local copy of something already in object
        storage, so it is always safe to drop; a later stage just re-downloads.
        Called after a successful render - before this existed, every job left
        60-150 MB behind forever.
        """
        size = sum(f.stat().st_size for f in self.root.rglob("*") if f.is_file())
        shutil.rmtree(self.root, ignore_errors=True)
        log.info("scratch cleaned: freed %.1f MB", size / 1_048_576)

    def __enter__(self) -> JobWorkspace:
        return self

    def __exit__(self, *exc) -> None:  # noqa: ANN002
        # Scratch is intentionally kept: it doubles as a cache between the
        # separate HTTP calls that n8n makes for each pipeline stage.
        return None
