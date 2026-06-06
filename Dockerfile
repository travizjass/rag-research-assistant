# syntax=docker/dockerfile:1

# ---- Stage 1: grab the Qdrant server binary from the official image ----
# We run Qdrant co-located with the app (single container) so the whole
# thing deploys as one free-tier web service. See start.sh.
FROM qdrant/qdrant:latest AS qdrant

# ---- Stage 2: the Python app (FastAPI + LangGraph + local embeddings) ----
FROM python:3.11-slim

# curl: used by start.sh to wait for Qdrant's health endpoint.
# ca-certificates: HTTPS to Groq / LangFuse / HuggingFace.
# libunwind8: runtime dependency of the Qdrant binary copied from stage 1
#             (provides libunwind-ptrace / libunwind-<arch> on amd64 and arm64).
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates libunwind8 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the CPU-only build of PyTorch FIRST. The default PyPI torch wheel
# bundles CUDA (~2.5 GB); the CPU index keeps the image small (~200 MB).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Python dependencies (sentence-transformers reuses the torch installed above).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model at build time so the first request is fast
# (otherwise ~90 MB is fetched on the first cold start).
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Embedded Qdrant server binary, copied from the official image above.
COPY --from=qdrant /qdrant/qdrant /usr/local/bin/qdrant

# Application code, UI, sample documents, and the startup script.
COPY . .
RUN chmod +x start.sh

# The app talks to the co-located Qdrant over localhost (started in start.sh).
ENV QDRANT_HOST=localhost \
    QDRANT_PORT=6333 \
    COLLECTION_NAME=research_docs \
    QDRANT_STORAGE_PATH=/data/qdrant \
    PYTHONUNBUFFERED=1

EXPOSE 8000

# start.sh: boot Qdrant -> wait healthy -> ingest sample docs (once) -> uvicorn.
CMD ["./start.sh"]
