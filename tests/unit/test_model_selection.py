"""
Model picker: allowlist enforcement, per-model caching, and the thinking-block
regression that made Sonnet 5 and Opus 5 unusable on the /query path.
"""

from __future__ import annotations

import pytest

from src.generation.llm_client import LLMClient, _first_text
from src.generation.models import (
    SELECTABLE_MODELS,
    ModelNotAllowed,
    is_allowed,
    resolve,
)


# ── The regression ────────────────────────────────────────────────────────────

class _Block:
    def __init__(self, type_: str, text: str | None = None):
        self.type = type_
        if text is not None:
            self.text = text


class _ThinkingBlock:
    """A thinking block as the SDK returns it: a `type`, and no `.text`."""
    type = "thinking"
    thinking = "Let me reason about this..."


def test_first_text_skips_leading_thinking_block():
    # Sonnet 5 / Opus 5 shape. The old code did content[0].text and raised
    # AttributeError here, 500-ing every /query served by those models.
    blocks = [_ThinkingBlock(), _Block("text", "the answer")]
    assert _first_text(blocks) == "the answer"


def test_first_text_handles_plain_text_response():
    # Sonnet 4.6 shape -- must keep behaving exactly as before.
    assert _first_text([_Block("text", "the answer")]) == "the answer"


def test_first_text_returns_empty_when_no_text_block():
    assert _first_text([_ThinkingBlock()]) == ""


def test_complete_survives_thinking_block(monkeypatch):
    """End-to-end on LLMClient.complete(), not just the helper."""
    client = LLMClient.__new__(LLMClient)
    client._provider = "anthropic"
    client._model = "claude-opus-5"

    class _Usage:
        input_tokens = 11
        output_tokens = 22

    class _Resp:
        content = [_ThinkingBlock(), _Block("text", "grounded answer")]
        usage = _Usage()

    class _Messages:
        def create(self, **kwargs):
            return _Resp()

    class _Client:
        messages = _Messages()

    client._client = _Client()

    out = client.complete("sys", [{"role": "user", "content": "q"}])
    assert out.content == "grounded answer"
    assert out.input_tokens == 11
    assert out.model == "claude-opus-5"


# ── Allowlist ─────────────────────────────────────────────────────────────────

def test_allowlist_contains_the_three_chosen_models():
    assert {m.id for m in SELECTABLE_MODELS} == {
        "claude-sonnet-4-6",
        "claude-sonnet-5",
        "claude-opus-5",
    }


def test_fable_is_not_selectable():
    # It returns 400 on forced tool use, which /treatment-card requires.
    assert not is_allowed("claude-fable-5-1")


@pytest.mark.parametrize("bogus", [
    "gpt-4o",
    "claude-3-opus-20240229",
    "../../etc/passwd",
    "claude-opus-5; DROP TABLE",
    "CLAUDE-OPUS-5",
])
def test_resolve_rejects_anything_off_the_allowlist(bogus):
    with pytest.raises(ModelNotAllowed):
        resolve(bogus, "claude-sonnet-4-6")


@pytest.mark.parametrize("empty", [None, ""])
def test_resolve_falls_back_to_default(empty):
    assert resolve(empty, "claude-sonnet-4-6") == "claude-sonnet-4-6"


def test_resolve_returns_default_unvalidated():
    # An operator's GENERATION_MODEL must never become a startup crash,
    # even if it names something outside the picker's allowlist.
    assert resolve(None, "some-internal-model") == "some-internal-model"


def test_resolve_passes_through_allowlisted_model():
    assert resolve("claude-opus-5", "claude-sonnet-4-6") == "claude-opus-5"


# ── Deadlock regression ───────────────────────────────────────────────────────
#
# The first version of model_selection used one non-reentrant lock: _select
# took it and then called _client_for, which took the same lock. Selecting
# any non-default model hung forever -- which is to say, the feature hung
# whenever it was actually used. The earlier route test missed it because it
# monkeypatched _client_for away, so the nested acquisition never happened.
# These tests drive the real code path and fail, rather than hang, if the
# nesting comes back.

import threading as _threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _fake_app():
    return SimpleNamespace(state=SimpleNamespace())


def _run_with_timeout(fn, seconds=10):
    """Run fn in a thread; report whether it finished rather than blocking."""
    done: list = []
    err: list = []

    def target():
        try:
            done.append(fn())
        except BaseException as exc:  # noqa: BLE001 - surfaced to the test
            err.append(exc)

    t = _threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout=seconds)
    return (not t.is_alive()), (done[0] if done else None), (err[0] if err else None)


def test_selecting_a_non_default_model_does_not_deadlock():
    import src.api.model_selection as sel

    app = _fake_app()
    request = SimpleNamespace(app=app)

    with patch("src.generation.llm_client.LLMClient") as fake_client_cls:
        fake_client_cls.side_effect = lambda provider, model, api_key: MagicMock(
            model=model, provider=provider
        )
        finished, result, err = _run_with_timeout(
            lambda: sel._select(
                request,
                "claude-opus-5",
                MagicMock(model="claude-sonnet-4-6"),
                lambda client: SimpleNamespace(model=client.model),
            )
        )

    assert finished, "model selection deadlocked on a non-default model"
    assert err is None, f"model selection raised: {err!r}"
    assert result.model == "claude-opus-5"


def test_concurrent_selection_of_several_models_does_not_deadlock():
    """Two locks that are never nested must also survive real contention."""
    import src.api.model_selection as sel

    app = _fake_app()
    request = SimpleNamespace(app=app)
    default = MagicMock(model="claude-sonnet-4-6")
    results: list = []
    errors: list = []

    def build(client):
        return SimpleNamespace(model=client.model)

    def worker(model_id):
        try:
            results.append(sel._select(request, model_id, default, build).model)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    with patch("src.generation.llm_client.LLMClient") as fake_client_cls:
        fake_client_cls.side_effect = lambda provider, model, api_key: MagicMock(
            model=model, provider=provider
        )
        threads = [
            _threading.Thread(target=worker, args=(m,), daemon=True)
            for m in ["claude-opus-5", "claude-sonnet-5"] * 4
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not any(t.is_alive() for t in threads), "concurrent selection deadlocked"

    assert not errors, f"concurrent selection raised: {errors!r}"
    assert len(results) == 8
    assert set(results) == {"claude-opus-5", "claude-sonnet-5"}


def test_client_is_cached_per_model_not_rebuilt_per_request():
    import src.api.model_selection as sel

    app = _fake_app()
    with patch("src.generation.llm_client.LLMClient") as fake_client_cls:
        fake_client_cls.side_effect = lambda provider, model, api_key: MagicMock(
            model=model, provider=provider
        )
        first = sel._client_for(app, "claude-opus-5")
        second = sel._client_for(app, "claude-opus-5")

    assert first is second
    assert fake_client_cls.call_count == 1


# ── Empty-answer guard ────────────────────────────────────────────────────────
#
# On a thinking model a hard question can spend the whole max_tokens budget
# reasoning and return content == ['thinking'] with no text block. Measured in
# production on claude-opus-5 at max_tokens=2000: stop_reason max_tokens,
# 2000/2000 output tokens, zero visible characters. Without a guard that
# renders as a blank answer with no error.

def test_generator_reports_rather_than_returning_a_blank_answer():
    from src.generation.generator import ClinicalGenerator

    class _Resp:
        content = ""          # what _first_text returns for a thinking-only reply
        input_tokens = 5000
        output_tokens = 8000
        model = "claude-opus-5"

    llm = MagicMock()
    llm.complete.return_value = _Resp()
    llm.provider = "anthropic"

    chunk = MagicMock()
    chunk.relevance_score = 0.9
    chunk.metadata = {"study_design": "rct", "title": "T", "year": 2024}
    chunk.text = "evidence"

    gen = ClinicalGenerator(llm_client=llm)
    result = gen.generate("a very hard question", [chunk])

    assert result.answer.strip(), "must not return an empty answer"
    assert "output limit" in result.answer
    assert result.citations == []
    assert result.model_used == "claude-opus-5"
    # Not retried: a fresh call cannot resume the truncated turn.
    assert llm.complete.call_count == 1
