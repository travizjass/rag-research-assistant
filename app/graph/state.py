"""LangGraph state definition shared by every node in the pipeline."""

from typing import List, Optional

from typing_extensions import TypedDict

from app.schemas.models import QueryAnalysis, RetrievedChunk


class GraphState(TypedDict, total=False):
    """Mutable state threaded through the graph nodes.

    Each node returns a *partial* dict that LangGraph merges into this
    structure, so every key is optional (``total=False``).
    """

    # --- Inputs ---
    query: str  # the original user question
    top_k: int  # number of chunks to retrieve

    # --- Query analysis ---
    query_analysis: Optional[QueryAnalysis]
    search_query: str  # query string actually sent to the retriever

    # --- Retrieval + grading ---
    documents: List[RetrievedChunk]  # raw chunks from Qdrant
    graded_documents: List[RetrievedChunk]  # chunks kept after relevance grading
    grading_score: float  # fraction of retrieved chunks judged relevant

    # --- Generation + verification ---
    answer: str
    hallucination_check: str  # "grounded" | "ungrounded"
    sources: List[str]  # citation strings backing the final answer

    # --- Loop counters (drive the retry budgets) ---
    retrieval_attempts: int
    generation_attempts: int
