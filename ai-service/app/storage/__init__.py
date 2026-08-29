"""Storage factory."""
from __future__ import annotations

from functools import lru_cache

from app.core.config import settings
from app.storage.base import StorageBackend
from app.storage.layout import JobLayout


@lru_cache(maxsize=1)
def get_storage() -> StorageBackend:
    if settings.storage_backend == "minio":
        from app.storage.minio_backend import MinIOStorage

        return MinIOStorage()
    from app.storage.local_backend import LocalStorage

    return LocalStorage()


__all__ = ["StorageBackend", "JobLayout", "get_storage"]
