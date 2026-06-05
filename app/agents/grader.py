"""Relevance Grader agent — keeps only chunks that actually help the query."""

from app.config import get_llm
from app.graph.state import GraphState
from app.schemas.models import RelevanceGrade

_SYSTEM_PROMPT = (
    "You are a relevance grader. Decide whether the retrieved document chunk "
    "contains information useful for answering the user's question. Be strict: "
    "mark it relevant only if it directly helps answer the question."
)


def grade_documents(state: GraphState) -> dict:
    """Grade each retrieved chunk and drop the irrelevant ones.

    Produces ``graded_documents`` (the survivors) and ``grading_score`` (the
    fraction of retrieved chunks judged relevant). On a grading error the
    chunk is kept, biasing toward recall over silently losing context.
    """
    grader = get_llm().with_structured_output(RelevanceGrade)
    documents = state.get("documents", [])
    question = state["query"]

    relevant = []
    for chunk in documents:
        try:
            grade: RelevanceGrade = grader.invoke(
                [
                    ("system", _SYSTEM_PROMPT),
                    ("human", f"Question:\n{question}\n\nChunk:\n{chunk.text}"),
                ]
            )
            keep = grade.is_relevant
        except Exception:
            keep = True
        if keep:
            relevant.append(chunk)

    score = len(relevant) / len(documents) if documents else 0.0
    return {"graded_documents": relevant, "grading_score": round(score, 2)}
