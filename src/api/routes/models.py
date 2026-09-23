"""GET /models -- the generation models a caller may select."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from config.settings import get_settings
from src.generation.models import SELECTABLE_MODELS

router = APIRouter(tags=["models"])


@router.get("/models")
async def list_models(request: Request) -> dict[str, Any]:
    """
    Return the allowlist so the UI renders one source of truth rather than
    its own copy of the model list.

    Unauthenticated on purpose: it exposes only model names the operator
    chose to offer, which the UI needs before a key is entered.
    """
    settings = get_settings()
    default_gen = getattr(request.app.state, "generator", None)
    default_model = getattr(default_gen, "model", "") or settings.generation_model
    return {
        "default": default_model,
        "models": [
            {
                "id": m.id,
                "label": m.label,
                "provider": m.provider,
                "is_default": m.id == default_model,
            }
            for m in SELECTABLE_MODELS
        ],
    }
