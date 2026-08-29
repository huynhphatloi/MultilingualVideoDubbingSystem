"""Filesystem storage backend (STORAGE_BACKEND=local)."""
from __future__ import annotations

import logging
import shutil
from collections.abc import Iterator
from pathlib import Path

from app.core.config import settings
from app.core.errors import ArtifactNotFound, StorageError
from app.storage.base import StorageBackend

log = logging.getLogger(__name__)


class LocalStorage(StorageBackend):
    name = "local"

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root or settings.local_storage_root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ helpers ---
    def _path(self, key: str) -> Path:
        candidate = (self.root / key).resolve()
        if not str(candidate).startswith(str(self.root)):
            raise StorageError(f"Key escapes storage root: {key}")
        return candidate

    # ------------------------------------------------------------ writing ---
    def put_file(self, key: str, local_path, content_type=None) -> str:  # noqa: ANN001
        dst = self._path(key)
        dst.parent.mkdir(parents=True, exist_ok=True)
        src = Path(local_path)
        if src.resolve() != dst:
            shutil.copyfile(src, dst)
        log.debug("stored %s (%d bytes)", key, dst.stat().st_size)
        return key

    def put_bytes(self, key: str, data: bytes, content_type=None) -> str:  # noqa: ANN001
        dst = self._path(key)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        return key

    # ------------------------------------------------------------ reading ---
    def get_file(self, key: str, local_path) -> Path:  # noqa: ANN001
        src = self._path(key)
        if not src.exists():
            raise ArtifactNotFound(f"No such object: {key}", details={"key": key})
        dst = Path(local_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src != dst.resolve():
            shutil.copyfile(src, dst)
        return dst

    def get_bytes(self, key: str) -> bytes:
        src = self._path(key)
        if not src.exists():
            raise ArtifactNotFound(f"No such object: {key}", details={"key": key})
        return src.read_bytes()

    def stream(self, key: str, chunk_size: int = 1024 * 1024) -> Iterator[bytes]:
        src = self._path(key)
        if not src.exists():
            raise ArtifactNotFound(f"No such object: {key}", details={"key": key})
        with src.open("rb") as fh:
            while chunk := fh.read(chunk_size):
                yield chunk

    # ----------------------------------------------------------- metadata ---
    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def size(self, key: str) -> int:
        path = self._path(key)
        if not path.exists():
            raise ArtifactNotFound(f"No such object: {key}", details={"key": key})
        return path.stat().st_size

    def list(self, prefix: str) -> list[str]:
        base = self._path(prefix)
        if base.is_file():
            return [prefix]
        if not base.exists():
            return []
        return sorted(
            str(p.relative_to(self.root)).replace("\\", "/")
            for p in base.rglob("*")
            if p.is_file()
        )

    def delete(self, key: str) -> None:
        path = self._path(key)
        if path.exists():
            path.unlink()

    def delete_prefix(self, prefix: str) -> int:
        base = self._path(prefix)
        if not base.exists():
            return 0
        count = sum(1 for p in base.rglob("*") if p.is_file())
        shutil.rmtree(base)
        return count

    # ------------------------------------------------------------- access ---
    def url(self, key: str, expires_seconds: int | None = None) -> str:
        # Served back through the API so the browser never touches the FS.
        return f"/artifacts/{key}"
