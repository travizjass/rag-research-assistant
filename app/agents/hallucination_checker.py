"""Hallucination Checker agent — verifies the answer is grounded in context."""

from app.agents.generator import format_context
from app.config import get_llm
from app.graph.state import GraphState
from app.schemas.models import HallucinationGrade

_SYSTEM_PROMPT = (
    "You are a fact-verification agent. Determine whether EVERY claim in the "
    "answer is supported by the provided context. If any claim is unsupported "
    "or contradicted by the context, mark the answer as not grounded."
)


def check_hallucination(state: GraphState) -> dict:
    """Check the generated answer against its supporting context.

    Returns ``hallucination_check`` set to ``"grounded"`` or ``"ungrounded"``.
    On a checker error it defaults to ``"grounded"`` to avoid looping forever.
    """
    context = format_context(state.get("graded_documents", []))
    answer = state.get("answer", "")

    try:
        checker = get_llm().with_structured_output(HallucinationGrade)
        grade: HallucinationGrade = checker.invoke(
            [
                ("system", _SYSTEM_PROMPT),
                ("human", f"Context:\n{context}\n\nAnswer:\n{answer}"),
            ]
        )
        grounded = grade.grounded
    except Exception:
        grounded = True

    return {"hallucination_check": "grounded" if grounded else "ungrounded"}
