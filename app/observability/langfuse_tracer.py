"""LangFuse setup: a single client plus a LangChain callback handler.

This module degrades gracefully — if keys are missing or the SDK is
unavailable, callers simply receive ``None`` and tracing is skipped, so the
pipeline keeps working without observability configured.
"""

from functools import lru_cache

from app.config import settings


@lru_cache(maxsize=None)
def get_langfuse():
    """Return a configured LangFuse client, or ``None`` if unavailable.

    Constructing the client also configures the global singleton that the
    LangChain ``CallbackHandler`` picks up automatically (LangFuse v3).
    """
    if not (settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY):
        return None
    try:
        from langfuse import Langfuse

        return Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST,
        )
    except Exception:
        # Never let observability setup break the request path.
        return None


@lru_cache(maxsize=None)
def get_callback_handler():
    """Return a LangChain callback handler wired to LangFuse, or ``None``.

    Pass the result inside ``config={"callbacks": [handler]}`` when invoking
    the graph to trace every node, token count, latency, and decision.
    """
    if get_langfuse() is None:
        return None
    try:
        from langfuse.langchain import CallbackHandler

        return CallbackHandler()
    except Exception:
        return None
