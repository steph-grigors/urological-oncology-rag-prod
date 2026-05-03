"""
API key authentication — FastAPI Depends function.
"""

from __future__ import annotations

import hmac

from fastapi import Depends, HTTPException, Request

from config.settings import Settings, get_settings

# Paths that never require authentication
_EXEMPT = {"/health/live", "/health/ready"}


async def require_api_key(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> str:
    """
    Return the validated API key, or raise HTTP 401.

    Skips auth entirely when:
      - The path is in the exempt set (/health probes).
      - APP_ENV == "development" and no keys are configured.
    Keys are compared in constant time to prevent timing attacks.
    """
    if request.url.path in _EXEMPT:
        return ""

    # Development bypass — allows local testing without setting API_KEYS
    if settings.app_env == "development" and not settings.api_keys:
        return "dev"

    key = request.headers.get(settings.api_key_header, "")
    if not key:
        raise HTTPException(status_code=401, detail="Missing API key")

    for valid in settings.api_keys:
        if hmac.compare_digest(key.encode(), valid.encode()):
            return key

    raise HTTPException(status_code=401, detail="Invalid API key")


async def require_admin_api_key(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> str:
    """Require the ADMIN_API_KEY header for privileged operations (e.g. /eval/run).

    Falls back to regular API key validation when no admin key is configured
    (development convenience).
    """
    admin_key = settings.admin_api_key
    if not admin_key:
        return await require_api_key(request, settings)

    key = request.headers.get(settings.api_key_header, "")
    if not key:
        raise HTTPException(status_code=401, detail="Missing API key")

    if hmac.compare_digest(key.encode(), admin_key.encode()):
        return key

    raise HTTPException(status_code=403, detail="Admin access required")
