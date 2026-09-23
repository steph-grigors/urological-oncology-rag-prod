"""
LLM call logic with provider abstraction and citation verification.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from config.constants import MAX_ANSWER_TOKENS
from src.generation.citations import DOC_TAG_RE, strip_invalid_citations
from src.generation.confidence import ConfidenceGate, compute_confidence, gate
from src.generation.post_process import apply_regulatory_warnings
from src.generation.prompts import (
    FALLBACK_DISCLAIMER,
    FALLBACK_USER_TEMPLATE,
    LOW_CONFIDENCE_REFUSAL,
    SYSTEM_PROMPT,
    build_prompt,
    with_safety_core,
)
from src.observability.logging import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from src.generation.llm_client import LLMClient
    from src.retrieval.reranker import RankedChunk


@dataclass
class GenerationResult:
    answer: str
    citations: list[int]
    evidence_quality: str
    model_used: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    confidence_score: float = 0.0
    hallucinated_citations: list[int] = field(default_factory=list)
    latency_ms: float = 0.0


class ClinicalGenerator:
    def __init__(self, llm_client: "LLMClient | None" = None) -> None:
        self._llm = llm_client

    @property
    def model(self) -> str:
        return self._llm.model if self._llm is not None else ""

    @property
    def provider(self) -> str:
        return self._llm.provider if self._llm is not None else ""

    def generate(
        self,
        query: str,
        ranked_chunks: list["RankedChunk"],
        conversation_history: list[dict] | None = None,
        system_prompt: str | None = None,
        confidence_score: float | None = None,
    ) -> GenerationResult:
        # `confidence_score` comes from the retriever, which scores the full
        # reranked candidate set. Recomputing here would see only the chunks
        # that survived grading, which cannot score below the grading
        # threshold -- the floor that made the caveated band unreachable.
        if confidence_score is None:
            confidence_result = compute_confidence(ranked_chunks)
            score = confidence_result.score
        else:
            score = confidence_score
        confidence_gate = gate(score)
        # A caller-supplied system_prompt is an addition to the safety core,
        # never a replacement for it. Before this, passing one replaced the
        # entire prompt: scope limits, citation discipline and the
        # insufficient-evidence rule all disappeared with it.
        active_system_prompt = with_safety_core(
            system_prompt if system_prompt is not None else SYSTEM_PROMPT
        )
        # /query has no explicit `language` field (unlike /treatment-card) — callers
        # that want French answers (e.g. onco-review-app) signal it by supplying
        # their own French system_prompt, so detect it the same way as
        # card_generator._reclassify_intent does, rather than always assuming English.
        answer_language = "fr" if "français" in active_system_prompt.lower() else "en"

        if self._llm is None:
            return GenerationResult(
                answer="No LLM client configured.",
                citations=[],
                evidence_quality="unknown",
                model_used="",
                provider="",
                prompt_tokens=0,
                completion_tokens=0,
                confidence_score=score,
            )

        if confidence_gate == ConfidenceGate.REFUSED:
            fallback_messages = [
                {"role": "user", "content": FALLBACK_USER_TEMPLATE.format(question=query)}
            ]
            if conversation_history:
                fallback_messages = conversation_history[-10:] + fallback_messages
            start = time.monotonic()
            response = self._llm.complete(
                active_system_prompt, fallback_messages, max_tokens=MAX_ANSWER_TOKENS
            )
            latency_ms = (time.monotonic() - start) * 1000
            return GenerationResult(
                answer=apply_regulatory_warnings(
                    FALLBACK_DISCLAIMER + response.content, language=answer_language
                ),
                citations=[],
                evidence_quality="insufficient",
                model_used=response.model,
                provider=self._llm.provider,
                prompt_tokens=response.input_tokens,
                completion_tokens=response.output_tokens,
                confidence_score=score,
                latency_ms=latency_ms,
            )

        # The gate's own value, so "caveated" reaches the prompt as itself
        # rather than collapsing into "hedged".
        confidence_level = confidence_gate.value
        messages = build_prompt(query, ranked_chunks, confidence_level=confidence_level)

        # Prepend last 5 turns (10 messages) of conversation history
        if conversation_history:
            messages = conversation_history[-10:] + messages

        start = time.monotonic()
        response = self._llm.complete(active_system_prompt, messages, max_tokens=MAX_ANSWER_TOKENS)
        latency_ms = (time.monotonic() - start) * 1000

        if not response.content.strip():
            # The response carried no text. Measured cause: on a thinking model
            # a hard question can spend the whole max_tokens budget reasoning
            # and stop before emitting any visible text (stop_reason
            # max_tokens, content == ['thinking']).
            #
            # Deliberately not retried. A retry is a fresh request: the API is
            # stateless, thinking is not carried across calls, and assistant
            # prefill -- the only way to ask a model to continue its own
            # truncated turn -- returns 400 on every model in the picker. So a
            # retry would re-reason from nothing and likely hit the same wall
            # at the same cost. Say so instead of rendering a blank answer.
            logger.error(
                "Generation returned no text (model=%s, completion_tokens=%d) -- "
                "budget likely exhausted before any visible output",
                response.model,
                response.output_tokens,
            )
            return GenerationResult(
                answer=(
                    "The model reached its output limit before producing an answer. "
                    "This question is unusually broad — narrowing it, or lowering the "
                    "number of sources retrieved, should let it complete."
                ),
                citations=[],
                evidence_quality=confidence_gate.value,
                model_used=response.model,
                provider=self._llm.provider,
                prompt_tokens=response.input_tokens,
                completion_tokens=response.output_tokens,
                confidence_score=score,
                latency_ms=latency_ms,
            )

        answer, hallucinated = self._check_citations(response.content, len(ranked_chunks))
        if hallucinated:
            answer = (
                "WARNING: The following answer contained hallucinated citations "
                f"([Doc {', '.join(str(n) for n in hallucinated)}]) that were removed.\n\n"
                + answer
            )

        citations = sorted(
            {int(m) for m in DOC_TAG_RE.findall(answer)} - set(hallucinated)
        )

        answer = apply_regulatory_warnings(answer, language=answer_language)

        return GenerationResult(
            answer=answer,
            citations=citations,
            evidence_quality=confidence_gate.value,
            model_used=response.model,
            provider=self._llm.provider,
            prompt_tokens=response.input_tokens,
            completion_tokens=response.output_tokens,
            confidence_score=score,
            hallucinated_citations=hallucinated,
            latency_ms=latency_ms,
        )

    @staticmethod
    def _check_citations(answer: str, num_docs: int) -> tuple[str, list[int]]:
        """Strip citations referencing non-existent docs; return (cleaned_answer, hallucinated_list)."""
        return strip_invalid_citations(answer, num_docs)
