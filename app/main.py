"""Application entry point — creates the FastAPI app and mounts the routes.

Run with:  uvicorn app.main:app --reload
Then open the UI at:  http://localhost:8000/ui/
"""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import router

# The ASGI application object referenced by ``uvicorn app.main:app``.
app = FastAPI(
    title="Multi-Agent RAG Research Assistant",
    description=(
        "A LangGraph RAG pipeline with relevance grading, hallucination "
        "detection, and LangFuse observability."
    ),
    version="1.0.0",
)

# Allow the browser UI (and any local tooling) to call the API freely.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount all API routes (e.g. POST /query, POST /query/stream, GET /health).
app.include_router(router)

# Serve the static web UI from the project-level ``ui/`` directory at /ui.
_UI_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ui"
)
if os.path.isdir(_UI_DIR):
    app.mount("/ui", StaticFiles(directory=_UI_DIR, html=True), name="ui")


@app.get("/")
def root() -> dict:
    """Return a tiny banner pointing at the UI and docs."""
    return {
        "service": "rag-research-assistant",
        "ui": "/ui/",
        "docs": "/docs",
        "query_endpoint": "POST /query",
        "stream_endpoint": "POST /query/stream",
    }
