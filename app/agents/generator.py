"""Generator agent — writes the answer grounded in the graded context."""

from typing import List

from app.config import get_llm
from app.graph.state import GraphState
from app.schemas.models import RetrievedChunk

_SYSTEM_PROMPT = (
    "You are a research assistant. Answer the user's question using ONLY the "
    "provided context. Reference the bracketed source numbers where relevant. "
    "If the context is insufficient, say so plainly rather than inventing facts."
)


def format_context(chunks: List[RetrievedChunk]) -> str:
    """Render graded chunks into a numbered, citation-tagged context block."""
    if not chunks:
        return "No context available."
    blocks = [
        f"[{i}] ({chunk.citation()})\n{chunk.text}"
        for i, chunk in enumerate(chunks, start=1)
    ]
    return "\n\n".join(blocks)


def generate(state: GraphState) -> dict:
    """Synthesise an answer from the graded context.

    Returns the ``answer``, the de-duplicated ``sources`` list, and an
    incremented ``generation_attempts`` counter that drives the retry budget.
    """
    chunks = state.get("graded_documents", [])
    context = format_context(chunks)
    # A touch of temperature keeps phrasing natural without harming grounding.
    llm = get_llm(temperature=0.2)

    answer = llm.invoke(
        [
            ("system", _SYSTEM_PROMPT),
            ("human", f"Context:\n{context}\n\nQuestion:\n{state['query']}"),
        ]
    ).content

    # Preserve order while removing duplicate citations.
    sources = list(dict.fromkeys(chunk.citation() for chunk in chunks))
    return {
        "answer": answer,
        "sources": sources,
        "generation_attempts": state.get("generation_attempts", 0) + 1,
    }
