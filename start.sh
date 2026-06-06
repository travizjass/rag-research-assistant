#!/usr/bin/env bash
# Container entrypoint: run an embedded Qdrant alongside the FastAPI app.
#
#   1. start the Qdrant server in the background
#   2. wait until it reports healthy
#   3. ingest the sample documents — but only if the collection is empty,
#      so restarts (with a persistent disk) don't create duplicate points
#   4. exec uvicorn on the platform-provided $PORT (Render/Railway set this)
set -euo pipefail

PORT="${PORT:-8000}"
COLLECTION_NAME="${COLLECTION_NAME:-research_docs}"
STORAGE_PATH="${QDRANT_STORAGE_PATH:-/data/qdrant}"

mkdir -p "$STORAGE_PATH"

echo "[start] launching embedded Qdrant (storage: $STORAGE_PATH) ..."
QDRANT__STORAGE__STORAGE_PATH="$STORAGE_PATH" \
QDRANT__SERVICE__HTTP_PORT=6333 \
  qdrant &
QDRANT_PID=$!

echo "[start] waiting for Qdrant to become healthy ..."
until curl -sf http://localhost:6333/healthz >/dev/null 2>&1; do
  if ! kill -0 "$QDRANT_PID" 2>/dev/null; then
    echo "[start] ERROR: Qdrant exited before becoming healthy" >&2
    exit 1
  fi
  sleep 1
done
echo "[start] Qdrant is up."

# Only ingest when the collection has no points yet (idempotent across restarts).
COUNT=$(COLLECTION_NAME="$COLLECTION_NAME" python - <<'PY'
import json, os, urllib.request
name = os.environ.get("COLLECTION_NAME", "research_docs")
try:
    with urllib.request.urlopen(f"http://localhost:6333/collections/{name}", timeout=5) as r:
        print(json.load(r)["result"]["points_count"] or 0)
except Exception:
    print(0)
PY
)
echo "[start] existing points in '$COLLECTION_NAME': $COUNT"

if [ "${COUNT:-0}" -eq 0 ] 2>/dev/null; then
  echo "[start] ingesting sample documents ..."
  python -m ingest.document_loader --path ./sample_documents/ \
    || echo "[start] WARN: ingestion failed (continuing; the UI will still load)"
else
  echo "[start] documents already present — skipping ingestion."
fi

echo "[start] launching FastAPI (uvicorn) on 0.0.0.0:$PORT ..."
exec uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
