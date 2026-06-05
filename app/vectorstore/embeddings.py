"""Sentence-Transformers embedding wrapper used for indexing and querying."""

from functools import lru_cache
from typing import List, Optional

from app.config import settings


class EmbeddingModel:
    """Thin wrapper around a SentenceTransformer model.

    The heavy model is loaded lazily on first use so that simply importing
    this module (for example during tests) stays cheap.
    """

    def __init__(self, model_name: Optional[str] = None):
        """Store the model name and defer the actual model load until needed."""
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self._model = None

    @property
    def model(self):
        """Load and cache the underlying SentenceTransformer on first access."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        """Return the embedding vector size (384 for all-MiniLM-L6-v2)."""
        return self.model.get_sentence_embedding_dimension()

    def embed_text(self, text: str) -> List[float]:
        """Embed a single string into one normalised float vector."""
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of strings; normalising makes cosine == dot product."""
        vectors = self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return [vector.tolist() for vector in vectors]


@lru_cache(maxsize=None)
def get_embedding_model() -> EmbeddingModel:
    """Return a process-wide singleton embedding model."""
    return EmbeddingModel()
