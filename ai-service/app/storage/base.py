"""Storage abstraction so the pipeline never cares about MinIO vs local disk."""
from __future__ import annotations

import abc
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


class StorageBackend(abc.ABC):
    """Object-store-ish interface. Keys look like ``jobs/<id>/audio/speech.wav``."""

    name: str = "base"

    # ------------------------------------------------------------ writing ---
    @abc.abstractmethod
    def put_file(self, key: str, local_path: str | Path, content_type: str | None = None) -> str:
        """Upload a local file. Returns the key."""

    @abc.abstractmethod
    def put_bytes(self, key: str, data: bytes, content_type: str | None = None) -> str:
        ...

    def put_json(self, key: str, payload: Any) -> str:
        raw = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        return self.put_bytes(key, raw, content_type="application/json")

    # ------------------------------------------------------------ reading ---
    @abc.abstractmethod
    def get_file(self, key: str, local_path: str | Path) -> Path:
        """Download to ``local_path``. Returns the local path."""

    @abc.abstractmethod
    def get_bytes(self, key: str) -> bytes:
        ...

    def get_json(self, key: str) -> Any:
        return json.loads(self.get_bytes(key).decode("utf-8"))

    @abc.abstractmethod
    def stream(self, key: str, chunk_size: int = 1024 * 1024) -> Iterator[bytes]:
        ...

    # ----------------------------------------------------------- metadata ---
    @abc.abstractmethod
    def exists(self, key: str) -> bool:
        ...

    @abc.abstractmethod
    def size(self, key: str) -> int:
        ...

    @abc.abstractmethod
    def list(self, prefix: str) -> list[str]:
        ...

    @abc.abstractmethod
    def delete(self, key: str) -> None:
        ...

    @abc.abstractmethod
    def delete_prefix(self, prefix: str) -> int:
        ...

    # ------------------------------------------------------------- access ---
    @abc.abstractmethod
    def url(self, key: str, expires_seconds: int | None = None) -> str:
        """A URL a browser can open (presigned for MinIO, API route for local)."""

    def content_type_for(self, key: str) -> str:
        ext = key.rsplit(".", 1)[-1].lower()
        return {
            "mp4": "video/mp4",
            "mkv": "video/x-matroska",
            "webm": "video/webm",
            "mov": "video/quicktime",
            "wav": "audio/wav",
            "mp3": "audio/mpeg",
            "flac": "audio/flac",
            "json": "application/json",
            "srt": "application/x-subrip",
            "vtt": "text/vtt",
            "txt": "text/plain",
        }.get(ext, "application/octet-stream")
