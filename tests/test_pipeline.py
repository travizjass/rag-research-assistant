"""Unit tests for the graph routing logic and core schemas.

These tests mock out all network / LLM / vector-store calls so they run fully
offline. Execute with:  python -m unittest tests.test_pipeline   (or: pytest)
"""

import unittest

from app.graph.pipeline import decide_after_check, decide_after_grading
from app.schemas.models import QueryRequest, QueryResponse, RetrievedChunk


class TestGradingRouter(unittest.TestCase):
    """Routing decisions taken after the relevance grader."""

    def test_generates_when_relevant_chunks_exist(self):
        """With graded docs present, the graph should advance to generation."""
        state = {"graded_documents": [RetrievedChunk(text="x")], "retrieval_attempts": 1}
        self.assertEqual(decide_after_grading(state), "generate")

    def test_retries_when_no_relevant_chunks(self):
        """With no graded docs and budget left, it should retry retrieval."""
        state = {"graded_documents": [], "retrieval_attempts": 1}
        self.assertEqual(decide_after_grading(state), "retry_retrieval")

    def test_gives_up_after_budget(self):
        """Once the retry budget is exhausted it should generate anyway."""
        state = {"graded_documents": [], "retrieval_attempts": 99}
        self.assertEqual(decide_after_grading(state), "generate")


class TestCheckRouter(unittest.TestCase):
    """Routing decisions taken after the hallucination checker."""

    def test_ends_when_grounded(self):
        """A grounded answer should terminate the graph."""
        state = {"hallucination_check": "grounded", "generation_attempts": 1}
        self.assertEqual(decide_after_check(state), "end")

    def test_regenerates_when_ungrounded(self):
        """An ungrounded answer with budget left should regenerate."""
        state = {"hallucination_check": "ungrounded", "generation_attempts": 1}
        self.assertEqual(decide_after_check(state), "regenerate")

    def test_ends_after_budget(self):
        """An ungrounded answer past its budget should end to avoid looping."""
        state = {"hallucination_check": "ungrounded", "generation_attempts": 99}
        self.assertEqual(decide_after_check(state), "end")


class TestSchemas(unittest.TestCase):
    """Validation and helper behaviour of the Pydantic models."""

    def test_query_request_defaults(self):
        """``top_k`` defaults to 5 when omitted."""
        self.assertEqual(QueryRequest(query="hello").top_k, 5)

    def test_citation_with_page(self):
        """A chunk with a page renders as ``source, page N``."""
        chunk = RetrievedChunk(text="t", source="doc.pdf", page=3)
        self.assertEqual(chunk.citation(), "doc.pdf, page 3")

    def test_citation_without_page(self):
        """A chunk without a page renders as just the source name."""
        chunk = RetrievedChunk(text="t", source="notes.txt")
        self.assertEqual(chunk.citation(), "notes.txt")

    def test_response_round_trip(self):
        """The response model serialises its fields faithfully."""
        resp = QueryResponse(
            answer="a",
            sources=["s"],
            grading_score=0.9,
            hallucination_check="grounded",
            retry_count=1,
        )
        self.assertEqual(resp.model_dump()["hallucination_check"], "grounded")


if __name__ == "__main__":
    unittest.main()
