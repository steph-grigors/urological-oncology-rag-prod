"""
Server-side allowlist of user-selectable generation models.

A request may name the model that answers it, but only from this fixed set.
Anything else is rejected before it reaches a provider SDK, so a caller can
never steer spend toward an arbitrary model string.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    id: str
    label: str
    provider: str


# Verified against the live Anthropic API on 2026-09-22: each model here
# returns a tool_use block under tool_choice={"type": "any"}, which the
# treatment card depends on, and each survives LLMClient.complete().
#
# claude-fable-5-1 is deliberately absent: it returns 400 on forced tool
# use, so it would serve /query but break /treatment-card. A model that
# works on one endpoint and not the other has no place in a picker that
# spans both.
SELECTABLE_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        id="claude-sonnet-4-6",
        label="Claude Sonnet 4.6",
        provider="anthropic",
    ),
    ModelSpec(
        id="claude-sonnet-5",
        label="Claude Sonnet 5",
        provider="anthropic",
    ),
    ModelSpec(
        id="claude-opus-5",
        label="Claude Opus 5",
        provider="anthropic",
    ),
)

_BY_ID: dict[str, ModelSpec] = {m.id: m for m in SELECTABLE_MODELS}


class ModelNotAllowed(ValueError):
    """Raised when a request names a model outside the allowlist."""


def is_allowed(model_id: str) -> bool:
    return model_id in _BY_ID


def get_spec(model_id: str) -> ModelSpec:
    try:
        return _BY_ID[model_id]
    except KeyError:
        raise ModelNotAllowed(
            f"Model {model_id!r} is not selectable. "
            f"Allowed: {', '.join(sorted(_BY_ID))}"
        ) from None


def resolve(requested: str | None, default: str) -> str:
    """
    Return the model that should answer this request.

    `requested` of None or "" means the caller expressed no preference and
    gets `default`. The default is returned unvalidated and on purpose: it
    comes from GENERATION_MODEL, which an operator sets deliberately, and
    validating it here would turn an operator's config choice into a
    startup crash. Only caller-supplied values are checked.
    """
    if not requested:
        return default
    if not is_allowed(requested):
        raise ModelNotAllowed(
            f"Model {requested!r} is not selectable. "
            f"Allowed: {', '.join(sorted(_BY_ID))}"
        )
    return requested
