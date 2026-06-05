"""Central configuration and shared client factories for the RAG assistant.

Every module that needs an API key, a host name, or a model id reads it from
here, so the rest of the codebase never touches ``os.environ`` directly.
"""

import os
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv

# Load variables from a local .env file (if present) into the process env.
load_dotenv()


class Settings:
    """Typed, single-source view over the environment variables we rely on."""

    # --- Groq / LLM ---
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0"))

    # --- Embeddings ---
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # --- Qdrant vector store ---
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "research_docs")

    # --- LangFuse (observability) ---
    LANGFUSE_PUBLIC_KEY: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    LANGFUSE_SECRET_KEY: str = os.getenv("LANGFUSE_SECRET_KEY", "")
    # README uses LANGFUSE_HOST while the local .env uses LANGFUSE_BASE_URL —
    # accept either spelling so both setups work unchanged.
    LANGFUSE_HOST: str = (
        os.getenv("LANGFUSE_HOST")
        or os.getenv("LANGFUSE_BASE_URL")
        or "https://cloud.langfuse.com"
    )

    # --- Retry budgets for the two graph loops ---
    MAX_RETRIEVAL_RETRIES: int = int(os.getenv("MAX_RETRIEVAL_RETRIES", "2"))
    MAX_GENERATION_RETRIES: int = int(os.getenv("MAX_GENERATION_RETRIES", "2"))


# Importable singleton used throughout the app.
settings = Settings()


@lru_cache(maxsize=None)
def get_llm(temperature: Optional[float] = None):
    """Return a cached Groq chat model (Llama 3.3-70B by default).

    The client is memoised per-temperature so every agent reuses one HTTP
    session instead of constructing a fresh model on each node execution.
    """
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=settings.LLM_MODEL,
        api_key=settings.GROQ_API_KEY,
        temperature=settings.LLM_TEMPERATURE if temperature is None else temperature,
    )
