"""Query Analyzer agent — decomposes and classifies the incoming question."""

from app.config import get_llm
from app.graph.state import GraphState
from app.schemas.models import QueryAnalysis

# Instructs the LLM to behave as a query-understanding component.
_SYSTEM_PROMPT = (
    "You are a query-analysis agent for a research assistant. "
    "Classify the user's question, break it into atomic sub-questions when "
    "useful, and rewrite it into a single concise search query optimised for "
    "semantic retrieval over a document corpus."
)


def analyze_query(state: GraphState) -> dict:
    """Classify the query and produce an optimised retrieval string.

    Returns a partial state update carrying the structured ``query_analysis``
    and the ``search_query`` the retriever should use. Falls back to the raw
    query if the LLM call fails, so the pipeline never stalls here.
    """
    try:
        llm = get_llm().with_structured_output(QueryAnalysis)
        analysis: QueryAnalysis = llm.invoke(
            [
                ("system", _SYSTEM_PROMPT),
                ("human", state["query"]),
            ]
        )
        return {
            "query_analysis": analysis,
            "search_query": analysis.search_query or state["query"],
        }
    except Exception:
        return {"query_analysis": None, "search_query": state["query"]}
