"""
LLM call logic with provider abstraction and citation verification.

Supports both Anthropic (Claude) and OpenAI (GPT) as generation backends,
selected by `settings.generation_provider`.  Switching providers requires
only a config change, not code changes.

Anthropic-specific features used when provider == "anthropic":
    - Extended thinking (budget_tokens configurable per request).
    - Prompt caching for the system prompt (cache_control: ephemeral)
      to reduce latency and cost on repeated calls with the same system prompt.

Citation grounding check:
    After generation, a lightweight post-processing step verifies that each
    [Doc N] citation in the answer actually appears in the context block.
    Hallucinated citations (e.g., [Doc 7] when only 5 docs were provided)
    are stripped and logged as a warning.

Public API (to be implemented):
    class Generator:
        def __init__(
            self,
            settings: Settings,
            anthropic_client=None,
            openai_client=None,
        ): ...

        def generate(
            self,
            messages: list[dict],
            model: str | None = None,
            max_tokens: int = MAX_ANSWER_TOKENS,
            stream: bool = False,
        ) -> GenerationResult | Iterator[str]: ...

        def check_citations(
            self,
            answer: str,
            num_docs: int,
        ) -> tuple[str, list[int]]:
            Return (cleaned_answer, list_of_hallucinated_doc_numbers).

    GenerationResult(dataclass)
        answer: str
        model: str
        provider: str
        input_tokens: int
        output_tokens: int
        latency_ms: float
        hallucinated_citations: list[int]   # doc numbers stripped
        cache_hit: bool                     # Anthropic prompt cache hit
"""
