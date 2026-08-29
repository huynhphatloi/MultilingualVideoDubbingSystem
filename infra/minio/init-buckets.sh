#!/bin/sh
# Idempotent MinIO bootstrap: alias, bucket, versioning, lifecycle.
set -eu

echo "[minio-init] waiting for MinIO..."
until mc alias set local http://minio:9000 "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD" >/dev/null 2>&1; do
  sleep 2
done

if mc ls "local/${MINIO_BUCKET}" >/dev/null 2>&1; then
  echo "[minio-init] bucket '${MINIO_BUCKET}' already exists"
else
  mc mb "local/${MINIO_BUCKET}"
  echo "[minio-init] created bucket '${MINIO_BUCKET}'"
fi

# Keep the console useful for the demo: browsable, but not publicly writable.
mc anonymous set download "local/${MINIO_BUCKET}" || true

# Auto-expire job artifacts after 30 days so the disk does not fill up.
cat > /tmp/lifecycle.json <<JSON
{
  "Rules": [
    {
      "ID": "expire-jobs",
      "Status": "Enabled",
      "Filter": { "Prefix": "jobs/" },
      "Expiration": { "Days": 30 }
    }
  ]
}
JSON
mc ilm import "local/${MINIO_BUCKET}" < /tmp/lifecycle.json || true

echo "[minio-init] done"
