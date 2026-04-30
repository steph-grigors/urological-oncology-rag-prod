"""
API key authentication middleware.

Reads the API key from the `X-API-Key` header (configurable via
`settings.api_key_header`) and validates it against the `settings.api_keys`
list.

Behaviour:
    - /health/live and /health/ready are always exempt from auth (probes must
      not require credentials).
    - All other routes require a valid API key.
    - Invalid or missing key → 401 with a generic error message (never reveal
      whether the key exists but is wrong vs. not set at all).
    - Keys are compared in constant time to prevent timing attacks.

Implementation:
    FastAPI `Request`-based dependency (`Depends(require_api_key)`) rather
    than Starlette middleware, so Swagger UI still works in development
    (APP_ENV == "development" → auth is skipped).

Public API (to be implemented):
    async def require_api_key(
        request: Request,
        settings: Settings = Depends(get_settings),
    ) -> str:
        Return the validated API key string, or raise HTTPException(401).
"""
