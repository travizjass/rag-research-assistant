"""Qdrant vector-store wrapper: collection management, upsert, and search."""

import uuid
from functools import lru_cache
from typing import Any, Dict, List, Optional

from app.config import settings
from app.schemas.models import RetrievedChunk
from app.vectorstore.embeddings import get_embedding_model


class QdrantStore:
    """High-level interface over a single Qdrant collection."""

    def __init__(self, collection_name: Optional[str] = None):
        """Connect to Qdrant and remember which collection to operate on."""
        from qdrant_client import QdrantClient

        self.collection_name = collection_name or settings.COLLECTION_NAME
        self.client = QdrantClient(
            host=settings.QDRANT_HOST, port=settings.QDRANT_PORT
        )
        self.embedder = get_embedding_model()

    def ensure_collection(self) -> None:
        """Create the collection with cosine distance if it doesn't exist yet."""
        from qdrant_client.models import Distance, VectorParams

        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedder.dimension, distance=Distance.COSINE
                ),
            )

    def upsert(self, chunks: List[Dict[str, Any]]) -> int:
        """Embed and store a batch of ``{text, source, page}`` dicts.

        Returns the number of points written to the collection.
        """
        from qdrant_client.models import PointStruct

        if not chunks:
            return 0
        self.ensure_collection()
        vectors = self.embedder.embed_texts([c["text"] for c in chunks])
        points = [
            PointStruct(id=str(uuid.uuid4()), vector=vector, payload=chunk)
            for vector, chunk in zip(vectors, chunks)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)
        return len(points)

    def search(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        """Embed the query and return the ``top_k`` most similar chunks."""
        vector = self.embedder.embed_text(query)
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=top_k,
            with_payload=True,
        )
        results: List[RetrievedChunk] = []
        for point in response.points:
            payload = point.payload or {}
            results.append(
                RetrievedChunk(
                    text=payload.get("text", ""),
                    source=payload.get("source", "unknown"),
                    page=payload.get("page"),
                    score=point.score,
                )
            )
        return results


@lru_cache(maxsize=None)
def get_store() -> QdrantStore:
    """Return a process-wide singleton Qdrant store."""
    return QdrantStore()
