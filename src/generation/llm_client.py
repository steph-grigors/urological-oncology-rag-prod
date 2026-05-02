"""
Provider-agnostic LLM client for the generation layer.

Supports Anthropic (Claude) and OpenAI (GPT) backends.
Switching providers requires only a config change.
"""

from __future__ import annotations

from dataclasses import dataclass


class ConfigurationError(Exception):
    pass


@dataclass
class LLMResponse:
    content: str
    input_tokens: int
    output_tokens: int
    model: str


class LLMClient:
    """
    Thin wrapper around Anthropic and OpenAI SDKs with a uniform interface.

    Pass api_key directly (rather than reading from the environment) so that
    tests can inject a mock key without mutating os.environ.
    """

    def __init__(self, provider: str, model: str, api_key: str = "") -> None:
        self._provider = provider
        self._model = model

        if provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key)
        elif provider == "openai":
            import openai
            self._client = openai.OpenAI(api_key=api_key)
        else:
            raise ConfigurationError(f"Unknown provider: {provider!r}")

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model(self) -> str:
        return self._model

    def complete(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int = 800,
    ) -> LLMResponse:
        """Send a completion request and return a normalised LLMResponse."""
        if self._provider == "anthropic":
            resp = self._client.messages.create(
                model=self._model,
                system=system,
                messages=messages,
                max_tokens=max_tokens,
            )
            return LLMResponse(
                content=resp.content[0].text,
                input_tokens=resp.usage.input_tokens,
                output_tokens=resp.usage.output_tokens,
                model=self._model,
            )
        else:  # openai
            all_messages = [{"role": "system", "content": system}] + messages
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=all_messages,
                max_tokens=max_tokens,
            )
            return LLMResponse(
                content=resp.choices[0].message.content,
                input_tokens=resp.usage.prompt_tokens,
                output_tokens=resp.usage.completion_tokens,
                model=self._model,
            )
