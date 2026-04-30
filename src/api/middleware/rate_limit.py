"""
Per-API-key rate limiting middleware.

Limits requests to `settings.rate_limit_per_minute` per API key per minute.
Uses a sliding window counter backed by an in-process dict (suitable for
single-instance deployments) with a TODO to migrate to Redis for
multi-instance production deployments.

Behaviour:
    - Counter resets on a rolling 60-second window.
    - Rate-limited requests → 429 Too Many Requests with
      `Retry-After` header indicating seconds until the window resets.
    - /health/* routes are exempt.
    - If no API key is present (anonymous), uses client IP as the key
      and applies a stricter limit (10 req/min).

Headers added to every response:
    X-RateLimit-Limit       configured limit
    X-RateLimit-Remaining   requests remaining in current window
    X-RateLimit-Reset       epoch seconds when the window resets

Public API (to be implemented):
    class RateLimitMiddleware(BaseHTTPMiddleware):
        def __init__(self, app, settings: Settings): ...
        async def dispatch(self, request: Request, call_next): ...
"""
