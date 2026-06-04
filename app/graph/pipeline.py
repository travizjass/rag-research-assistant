"""Assembles the multi-agent RAG pipeline as a stateful LangGraph graph.

Flow:  query_analyzer -> retriever -> grader -(relevant?)-> generator
       -> hallucination_checker -(grounded?)-> END, with retry loops back to
       the retriever (on irrelevance) and the generator (on hallucination).
"""

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from app.agents.generator import generate
from app.agents.grader import grade_documents
from app.agents.hallucination_checker import check_hallucination
from app.agents.query_analyzer import analyze_query
from app.agents.retriever import retrieve
from app.config import settings
from app.graph.state import GraphState


def decide_after_grading(state: GraphState) -> str:
    """Route after relevance grading.

    Generate when relevant chunks survived; otherwise retry retrieval until the
    budget is spent, then generate with whatever context is available.
    """
    if state.get("graded_documents"):
        return "generate"
    if state.get("retrieval_attempts", 0) <= settings.MAX_RETRIEVAL_RETRIES:
        return "retry_retrieval"
    return "generate"


def decide_after_check(state: GraphState) -> str:
    """Route after the hallucination check.

    End when the answer is grounded; otherwise regenerate until the budget is
    spent, then end anyway to avoid an infinite loop.
    """
    if state.get("hallucination_check") == "grounded":
        return "end"
    if state.get("generation_attempts", 0) <= settings.MAX_GENERATION_RETRIES:
        return "regenerate"
    return "end"


def build_graph():
    """Wire every agent node and conditional edge into a compiled graph."""
    graph = StateGraph(GraphState)

    # Register each agent as a node.
    graph.add_node("query_analyzer", analyze_query)
    graph.add_node("retriever", retrieve)
    graph.add_node("grader", grade_documents)
    graph.add_node("generator", generate)
    graph.add_node("hallucination_checker", check_hallucination)

    # Linear backbone of the pipeline.
    graph.add_edge(START, "query_analyzer")
    graph.add_edge("query_analyzer", "retriever")
    graph.add_edge("retriever", "grader")

    # Retry retrieval if no relevant chunks were found.
    graph.add_conditional_edges(
        "grader",
        decide_after_grading,
        {"generate": "generator", "retry_retrieval": "retriever"},
    )

    graph.add_edge("generator", "hallucination_checker")

    # Re-generate if a hallucination was detected.
    graph.add_conditional_edges(
        "hallucination_checker",
        decide_after_check,
        {"end": END, "regenerate": "generator"},
    )

    return graph.compile()


@lru_cache(maxsize=None)
def get_graph():
    """Return a compiled graph singleton (built once, reused per request)."""
    return build_graph()
