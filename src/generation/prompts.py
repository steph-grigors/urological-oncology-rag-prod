"""
All prompt templates for the generation layer.

Keeping prompts in one module makes them easy to version, A/B test, and
audit without touching generation logic.

Prompt design principles for clinical decision support:
    - System prompt explicitly states the assistant's scope (urological
      oncology research summarisation) and hard limits (no diagnosis,
      no treatment decisions).
    - Source citation is mandatory: every factual claim must cite [Doc N].
    - Uncertainty must be expressed when evidence is conflicting or sparse.
    - MEDICAL_DISCLAIMER (from constants.py) is appended to every answer.
    - Temperature is kept at 0.1 to minimise hallucination.

Templates (to be implemented as string constants or Jinja2 templates):
    SYSTEM_PROMPT
        Role definition, scope constraints, citation instruction,
        clinical caveat, and output format specification.

    USER_PROMPT_TEMPLATE
        Slots: {context_block}, {question}
        Formats retrieved chunks as [Doc N] blocks with title/section/year.

    HEDGED_ANSWER_PREFIX
        Prepended when retrieval_confidence is between CONFIDENCE_LOW and
        CONFIDENCE_HIGH.  Signals to the reader that evidence is limited.

    LOW_CONFIDENCE_REFUSAL
        Full refusal text returned when retrieval_confidence < CONFIDENCE_REFUSE.

    QUERY_REWRITE_PROMPT
        Used by ConversationMemory to expand follow-up questions into
        standalone queries using prior turn context.

Public API (to be implemented):
    build_prompt(
        question: str,
        chunks: list[SearchResult],
        max_context_chars: int,
        confidence_level: Literal["high", "hedged"],
    ) -> list[dict]
        Return the messages list ready for the LLM client.

    format_context_block(chunks: list[SearchResult], max_chars: int) -> str
        Format chunks as numbered [Doc N] blocks, truncating to max_chars.
"""
