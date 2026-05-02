"""
Structured JSON logging.
"""

from __future__ import annotations

import contextvars
import json
import logging
from datetime import datetime, timezone

# Context variables set by the request-ID middleware and query handler
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")
query_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("query_id", default="")


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_var.get(""),
            "query_id": query_id_var.get(""),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


class RequestContextFilter(logging.Filter):
    """Inject request_id and query_id context vars into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get("")  # type: ignore[attr-defined]
        record.query_id = query_id_var.get("")  # type: ignore[attr-defined]
        return True


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger with JSON output. Call once at startup."""
    handler = logging.StreamHandler()
    handler.setFormatter(_JsonFormatter())
    handler.addFilter(RequestContextFilter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))


def get_logger(name: str) -> logging.Logger:
    """Return a pre-configured logger. Use instead of logging.getLogger()."""
    return logging.getLogger(name)
