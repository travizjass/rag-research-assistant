"""FastAPI routes exposing the RAG pipeline over HTTP.

Two flavours of the same pipeline:
  * POST /query         -> runs the graph and returns the final JSON answer.
  * POST /query/stream  -> runs the graph and streams each agent's step live
                           as Server-Sent Events (SSE), so a UI can show the
                           agent "thinking" node by node.
"""

import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.graph.pipeline import get_graph
from app.observability.langfuse_tracer import get_callback_handler
from app.schemas.models import QueryRequest, QueryResponse

router = APIRouter()

# Human-friendly labels for each pipeline node, used by the streaming UI.
NODE_LABELS = {
    "query_analyzer": "Query Analyzer",
    "retriever": "Retriever",
    "grader": "Relevance Grader",
    "generator": "Generator",
    "hallucination_checker": "Hallucination Checker",
}


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Run a question through the full multi-agent RAG pipeline (non-streaming).

    Threads the query through analyzer -> retriever -> grader -> generator ->
    hallucination checker and returns the grounded answer with its sources.
    """
    graph = get_graph()

    # Attach the LangFuse tracer (if configured) so the whole run is observable.
    handler = get_callback_handler()
    config = {"callbacks": [handler]} if handler else {}

    try:
        result = graph.invoke(
            {"query": request.query, "top_k": request.top_k},
            config=config,
        )
    except Exception as exc:  # surface any pipeline failure as a clean 500
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    retry_count = max(0, result.get("retrieval_attempts", 1) - 1) + max(
        0, result.get("generation_attempts", 1) - 1
    )

    return QueryResponse(
        answer=result.get("answer", ""),
        sources=result.get("sources", []),
        grading_score=result.get("grading_score", 0.0),
        hallucination_check=result.get("hallucination_check", "unknown"),
        retry_count=retry_count,
    )


def _chunk_summary(chunk) -> dict:
    """Serialise a RetrievedChunk into a small JSON-safe dict for the UI."""
    snippet = chunk.text.strip().replace("\n", " ")
    if len(snippet) > 220:
        snippet = snippet[:220] + "..."
    return {
        "source": chunk.source,
        "page": chunk.page,
        "score": round(chunk.score, 3),
        "snippet": snippet,
    }


def _describe(node: str, update: dict) -> dict:
    """Turn a node's raw state update into a human-readable event payload."""
    if node == "query_analyzer":
        analysis = update.get("query_analysis")
        return {
            "search_query": update.get("search_query"),
            "query_type": analysis.query_type.value if analysis else None,
            "sub_queries": list(analysis.sub_queries) if analysis else [],
        }
    if node == "retriever":
        docs = update.get("documents", []) or []
        return {
            "attempt": update.get("retrieval_attempts"),
            "retrieved": len(docs),
            "documents": [_chunk_summary(d) for d in docs],
        }
    if node == "grader":
        docs = update.get("graded_documents", []) or []
        return {
            "grading_score": update.get("grading_score"),
            "kept": len(docs),
            "documents": [_chunk_summary(d) for d in docs],
        }
    if node == "generator":
        return {
            "answer": update.get("answer"),
            "sources": update.get("sources", []) or [],
            "attempt": update.get("generation_attempts"),
        }
    if node == "hallucination_checker":
        return {"hallucination_check": update.get("hallucination_check")}
    return {}


def _sse(event_type: str, payload: dict) -> str:
    """Format a payload as a single Server-Sent Events ``data:`` frame."""
    return f"data: {json.dumps({'type': event_type, **payload})}\n\n"


@router.post("/query/stream")
def query_stream(request: QueryRequest) -> StreamingResponse:
    """Run the pipeline and stream every agent's step as it completes.

    Emits SSE frames: one ``start``, a ``step`` per node execution (retries
    included), a final ``done`` with the consolidated result, or an ``error``.
    """
    graph = get_graph()
    handler = get_callback_handler()
    config = {"callbacks": [handler]} if handler else {}

    def event_stream():
        """Generator that yields SSE frames as the graph streams updates."""
        # Track the latest value of each field to build the final result.
        final = {
            "answer": "",
            "sources": [],
            "grading_score": 0.0,
            "hallucination_check": "unknown",
            "retrieval_attempts": 0,
            "generation_attempts": 0,
        }
        try:
            yield _sse("start", {"query": request.query, "top_k": request.top_k})

            # stream_mode="updates" yields {node_name: partial_state} after each node.
            for update in graph.stream(
                {"query": request.query, "top_k": request.top_k},
                config=config,
                stream_mode="updates",
            ):
                for node, node_update in update.items():
                    node_update = node_update or {}
                    # Accumulate the fields we need for the final summary.
                    for key in final:
                        if node_update.get(key) is not None:
                            final[key] = node_update[key]
                    yield _sse(
                        "step",
                        {
                            "node": node,
                            "label": NODE_LABELS.get(node, node),
                            "detail": _describe(node, node_update),
                        },
                    )

            retry_count = max(0, final["retrieval_attempts"] - 1) + max(
                0, final["generation_attempts"] - 1
            )
            yield _sse(
                "done",
                {
                    "answer": final["answer"],
                    "sources": final["sources"],
                    "grading_score": final["grading_score"],
                    "hallucination_check": final["hallucination_check"],
                    "retry_count": retry_count,
                },
            )
        except Exception as exc:
            yield _sse("error", {"message": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/health")
def health() -> dict:
    """Lightweight liveness probe for load balancers and uptime checks."""
    return {"status": "ok"}
