"""
API key authentication — FastAPI Depends function.
"""

from __future__ import annotations

import hashlib
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


def api_key_fingerprint(api_key: str) -> str | None:
    """Return a stable, non-reversible identifier for an API key.

    The audit log needs to attribute a query to a caller, but storing the key
    itself turns `audit_log.user_id` into a credential store: anyone who can
    read that table can then call the API as that caller. A truncated SHA-256
    keeps both attribution and grouping -- the same key always maps to the same
    fingerprint, so "which caller ran this query" and "all queries by caller X"
    still work -- without the stored value being usable to authenticate.

    Returns None for the unauthenticated and development sentinels, matching
    the previous behaviour of writing NULL for those.
    """
    if api_key in ("", "dev"):
        return None
    return f"k_{hashlib.sha256(api_key.encode('utf-8')).hexdigest()[:16]}"
