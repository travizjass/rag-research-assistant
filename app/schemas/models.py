"""Pydantic models shared across the API, the agents, and the graph state.

The README splits these into query/response/state/grading files; this project
keeps them in one cohesive module to avoid circular imports and churn.
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class QueryType(str, Enum):
    """Coarse category the Query Analyzer assigns to an incoming question."""

    FACTUAL = "factual"
    COMPARATIVE = "comparative"
    SUMMARIZATION = "summarization"
    EXPLORATORY = "exploratory"
    OTHER = "other"


class QueryRequest(BaseModel):
    """Incoming JSON payload for ``POST /query``."""

    query: str = Field(..., description="Natural-language research question.")
    top_k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve.")


class RetrievedChunk(BaseModel):
    """A single chunk of text returned from the vector store."""

    text: str
    source: str = "unknown"
    page: Optional[int] = None
    score: float = 0.0

    def citation(self) -> str:
        """Render a human-readable source label like ``doc.pdf, page 3``."""
        if self.page is not None:
            return f"{self.source}, page {self.page}"
        return self.source


class QueryAnalysis(BaseModel):
    """Structured output of the Query Analyzer agent."""

    query_type: QueryType = Field(
        QueryType.OTHER, description="Category of the question."
    )
    sub_queries: List[str] = Field(
        default_factory=list, description="Decomposed atomic sub-questions."
    )
    search_query: str = Field(
        ..., description="Concise query string optimised for semantic retrieval."
    )


class RelevanceGrade(BaseModel):
    """Per-chunk verdict produced by the Relevance Grader."""

    is_relevant: bool = Field(
        ..., description="True if the chunk helps answer the query."
    )
    reason: str = Field("", description="Short justification for the decision.")


class HallucinationGrade(BaseModel):
    """Verdict produced by the Hallucination Checker."""

    grounded: bool = Field(
        ..., description="True if the answer is supported by the context."
    )
    reason: str = Field("", description="Short justification for the decision.")


class QueryResponse(BaseModel):
    """Final JSON payload returned by ``POST /query``."""

    answer: str
    sources: List[str] = Field(default_factory=list)
    grading_score: float = 0.0
    hallucination_check: str = "unknown"
    retry_count: int = 0
