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
