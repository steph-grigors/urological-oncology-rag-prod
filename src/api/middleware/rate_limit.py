"""
In-process sliding-window rate limiter.

TODO: Replace the in-process dict with Redis for multi-instance deployments.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

if TYPE_CHECKING:
    from config.settings import Settings

_ANON_LIMIT = 10          # requests/minute for unauthenticated callers
_WINDOW_SECONDS = 60


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, settings: "Settings") -> None:
        super().__init__(app)
        self._limit = settings.rate_limit_per_minute
        self._api_key_header = settings.api_key_header
        # {counter_key: (count, window_start_epoch)}
        self._counters: dict[str, tuple[int, float]] = {}

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path.startswith("/health"):
            return await call_next(request)

        counter_key, limit = self._resolve_key(request)
        now = time.time()
        count, window_start = self._counters.get(counter_key, (0, now))

        # Roll the window forward when the last window has expired
        if now - window_start >= _WINDOW_SECONDS:
            count, window_start = 0, now

        remaining = max(0, limit - count - 1)
        reset_at = int(window_start + _WINDOW_SECONDS)

        if count >= limit:
            retry_after = max(1, int(window_start + _WINDOW_SECONDS - now) + 1)
            return Response(
                content=json.dumps({"detail": "Rate limit exceeded"}),
                status_code=429,
                media_type="application/json",
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_at),
                },
            )

        self._counters[counter_key] = (count + 1, window_start)
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_at)
        return response

    def _resolve_key(self, request: Request) -> tuple[str, int]:
        """Return (counter_key, limit) for this request."""
        api_key = request.headers.get(self._api_key_header, "")
        if api_key:
            return f"key:{api_key}", self._limit
        host = request.client.host if request.client else "unknown"
        return f"ip:{host}", _ANON_LIMIT
