"""
Langfuse distributed tracing — gracefully degrades to no-ops when not configured.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from config.settings import Settings

# Optional Langfuse import — no-op if package not installed or key not set
try:
    import langfuse as _langfuse_module
    _LANGFUSE_AVAILABLE = True
except ImportError:
    _langfuse_module = None  # type: ignore[assignment]
    _LANGFUSE_AVAILABLE = False

_client: Any = None


def setup_tracing(settings: "Settings") -> None:
    """Initialise the Langfuse client. Call once at startup."""
    global _client
    if not _LANGFUSE_AVAILABLE or not settings.langfuse_public_key:
        return
    _client = _langfuse_module.Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )


# ── No-op stubs used when Langfuse is disabled ────────────────────────────────

class _NullSpan:
    def __enter__(self) -> "_NullSpan":
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def end(self, **kwargs: Any) -> None:
        pass


class QueryTrace:
    """Wraps a Langfuse trace (or no-ops when tracing is disabled)."""

    def __init__(self, trace: Any = None) -> None:
        self._trace = trace

    @contextmanager
    def span(self, name: str, **kwargs: Any) -> Iterator[Any]:
        if self._trace is None:
            yield _NullSpan()
            return
        s = self._trace.span(name=name, **kwargs)
        try:
            yield s
        finally:
            s.end()

    def score(self, name: str, value: float, comment: str = "") -> None:
        if self._trace is not None:
            self._trace.score(name=name, value=value, comment=comment)

    def set_metadata(self, **kwargs: Any) -> None:
        if self._trace is not None:
            self._trace.update(metadata=kwargs)


@contextmanager
def trace_query(query_id: str, question: str) -> Iterator[QueryTrace]:
    """Open a Langfuse trace for one query, yielding a QueryTrace helper."""
    if _client is None:
        yield QueryTrace(None)
        return
    trace = _client.trace(id=query_id, input=question)
    try:
        yield QueryTrace(trace)
    finally:
        _client.flush()
