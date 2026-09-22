"""
Per-request generation model selection.

A request may name the model that answers it. The name is checked against
the allowlist in `src.generation.models`, then mapped to a generator built
on an LLMClient for that model.

Clients are cached per model rather than rebuilt per request, because each
LLMClient owns a provider SDK client with its own connection pool. The
cache needs no eviction policy: it is keyed by allowlisted model id, so it
can never hold more entries than the allowlist has members.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from config.settings import get_settings
from src.generation.models import ModelNotAllowed, get_spec, resolve

if TYPE_CHECKING:
    from fastapi import Request

# Two separate locks, never held at the same time. A single lock here
# deadlocked: _select held it and then called _client_for, which tried to
# take the same non-reentrant lock. Clients are therefore built outside the
# generator lock, and neither function calls the other while holding one.
_client_lock = threading.Lock()
_generator_lock = threading.Lock()


def _client_cache(app: Any) -> dict:
    cache = getattr(app.state, "model_clients", None)
    if cache is None:
        cache = {}
        app.state.model_clients = cache
    return cache


def _client_for(app: Any, model_id: str):
    """Return a cached LLMClient for `model_id`, building one if needed."""
    from src.generation.llm_client import LLMClient

    cache = _client_cache(app)
    client = cache.get(model_id)
    if client is not None:
        return client

    with _client_lock:
        # Re-check: another thread may have built it while we waited.
        client = cache.get(model_id)
        if client is not None:
            return client

        settings = get_settings()
        spec = get_spec(model_id)
        api_key = (
            settings.anthropic_api_key
            if spec.provider == "anthropic"
            else settings.openai_api_key
        )
        client = LLMClient(provider=spec.provider, model=model_id, api_key=api_key)
        cache[model_id] = client
        return client


def _select(request: "Request", requested: str | None, default_instance: Any, build):
    """
    Return the generator that should serve this request.

    When the caller named no model, or named the one the default instance
    already runs, the default instance is returned untouched -- so the
    common path behaves exactly as it did before model selection existed.
    """
    settings = get_settings()
    default_model = getattr(default_instance, "model", "") or settings.generation_model
    model_id = resolve(requested, default_model)
    if model_id == default_model:
        return default_instance

    cache = getattr(request.app.state, "model_generators", None)
    if cache is None:
        cache = {}
        request.app.state.model_generators = cache

    key = (build.__name__, model_id)
    gen = cache.get(key)
    if gen is not None:
        return gen

    # Built before taking the generator lock, never underneath it: nesting
    # the two is what deadlocked. _client_for does its own locking, so at
    # most one client per model is cached even if two threads race here.
    client = _client_for(request.app, model_id)

    with _generator_lock:
        gen = cache.get(key)
        if gen is None:
            gen = build(client)
            cache[key] = gen
        return gen


def generator_for(request: "Request", requested: str | None, default_instance: Any):
    from src.generation.generator import ClinicalGenerator

    def build_clinical_generator(client):
        return ClinicalGenerator(llm_client=client)

    return _select(request, requested, default_instance, build_clinical_generator)


def card_generator_for(request: "Request", requested: str | None, default_instance: Any):
    from src.generation.card_generator import CardGenerator

    def build_card_generator(client):
        return CardGenerator(llm_client=client)

    return _select(request, requested, default_instance, build_card_generator)


__all__ = ["generator_for", "card_generator_for", "ModelNotAllowed"]
