"""Retriever agent — fetches the top-k candidate chunks from Qdrant."""

from app.graph.state import GraphState
from app.vectorstore.qdrant_client import get_store


def retrieve(state: GraphState) -> dict:
    """Fetch candidate chunks for the current search query.

    On each retry loop the net is widened (``top_k`` grows) to surface chunks
    the previous, stricter pass may have missed. Also bumps the
    ``retrieval_attempts`` counter that drives the retry budget.
    """
    attempts = state.get("retrieval_attempts", 0)
    base_k = state.get("top_k", 5)
    # Widen the search a little on every subsequent attempt.
    k = base_k + attempts * 3
    query = state.get("search_query") or state["query"]

    store = get_store()
    documents = store.search(query, top_k=k)
    return {"documents": documents, "retrieval_attempts": attempts + 1}
