"""MinIO / S3 storage backend (STORAGE_BACKEND=minio)."""
from __future__ import annotations

import io
import logging
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

from app.core.config import settings
from app.core.errors import ArtifactNotFound, StorageError
from app.storage.base import StorageBackend

log = logging.getLogger(__name__)


class MinIOStorage(StorageBackend):
    name = "minio"

    def __init__(self) -> None:
        from minio import Minio

        self.bucket = settings.minio_bucket
        self._client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )
        # A separate client bound to the browser-visible host so presigned
        # signatures match the URL the user actually opens.
        self._public_client = Minio(
            settings.minio_public_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        try:
            if not self._client.bucket_exists(self.bucket):
                self._client.make_bucket(self.bucket)
                log.info("created bucket %s", self.bucket)
        except Exception as exc:  # pragma: no cover - connectivity issue
            raise StorageError(f"Cannot reach MinIO: {exc}") from exc

    # ------------------------------------------------------------ writing ---
    def put_file(self, key: str, local_path, content_type=None) -> str:  # noqa: ANN001
        path = Path(local_path)
        if not path.exists():
            raise StorageError(f"Local file missing: {path}")
        self._client.fput_object(
            self.bucket, key, str(path),
            content_type=content_type or self.content_type_for(key),
        )
        log.debug("uploaded %s (%d bytes)", key, path.stat().st_size)
        return key

    def put_bytes(self, key: str, data: bytes, content_type=None) -> str:  # noqa: ANN001
        self._client.put_object(
            self.bucket, key, io.BytesIO(data), length=len(data),
            content_type=content_type or self.content_type_for(key),
        )
        return key

    # ------------------------------------------------------------ reading ---
    def get_file(self, key: str, local_path) -> Path:  # noqa: ANN001
        dst = Path(local_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._client.fget_object(self.bucket, key, str(dst))
        except Exception as exc:
            raise ArtifactNotFound(f"No such object: {key}", details={"key": key}) from exc
        return dst

    def get_bytes(self, key: str) -> bytes:
        response = None
        try:
            response = self._client.get_object(self.bucket, key)
            return response.read()
        except Exception as exc:
            raise ArtifactNotFound(f"No such object: {key}", details={"key": key}) from exc
        finally:
            if response is not None:
                response.close()
                response.release_conn()

    def stream(self, key: str, chunk_size: int = 1024 * 1024) -> Iterator[bytes]:
        response = None
        try:
            response = self._client.get_object(self.bucket, key)
            while chunk := response.read(chunk_size):
                yield chunk
        except Exception as exc:
            raise ArtifactNotFound(f"No such object: {key}", details={"key": key}) from exc
        finally:
            if response is not None:
                response.close()
                response.release_conn()

    # ----------------------------------------------------------- metadata ---
    def exists(self, key: str) -> bool:
        try:
            self._client.stat_object(self.bucket, key)
            return True
        except Exception:
            return False

    def size(self, key: str) -> int:
        try:
            return self._client.stat_object(self.bucket, key).size
        except Exception as exc:
            raise ArtifactNotFound(f"No such object: {key}", details={"key": key}) from exc

    def list(self, prefix: str) -> list[str]:
        return sorted(
            obj.object_name
            for obj in self._client.list_objects(self.bucket, prefix=prefix, recursive=True)
        )

    def delete(self, key: str) -> None:
        try:
            self._client.remove_object(self.bucket, key)
        except Exception as exc:  # pragma: no cover
            log.warning("delete failed for %s: %s", key, exc)

    def delete_prefix(self, prefix: str) -> int:
        from minio.deleteobjects import DeleteObject

        keys = self.list(prefix)
        if not keys:
            return 0
        errors = list(
            self._client.remove_objects(self.bucket, (DeleteObject(k) for k in keys))
        )
        for err in errors:  # pragma: no cover
            log.warning("delete error: %s", err)
        return len(keys)

    # ------------------------------------------------------------- access ---
    def url(self, key: str, expires_seconds: int | None = None) -> str:
        expiry = timedelta(seconds=expires_seconds or settings.presign_expiry_seconds)
        try:
            return self._public_client.presigned_get_object(self.bucket, key, expires=expiry)
        except Exception as exc:  # pragma: no cover
            raise StorageError(f"Cannot presign {key}: {exc}") from exc
