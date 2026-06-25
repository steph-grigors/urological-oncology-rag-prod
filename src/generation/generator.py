"""
LLM call logic with provider abstraction and citation verification.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from config.constants import MAX_ANSWER_TOKENS
from src.generation.confidence import ConfidenceGate, compute_confidence, gate
from src.generation.post_process import apply_regulatory_warnings
from src.generation.prompts import (
    FALLBACK_DISCLAIMER,
    FALLBACK_USER_TEMPLATE,
    LOW_CONFIDENCE_REFUSAL,
    SYSTEM_PROMPT,
    build_prompt,
)

if TYPE_CHECKING:
    from src.generation.llm_client import LLMClient
    from src.retrieval.reranker import RankedChunk

_CITATION_RE = re.compile(r"\[Doc (\d+)\]")


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

    def generate(
        self,
        query: str,
        ranked_chunks: list["RankedChunk"],
        conversation_history: list[dict] | None = None,
        system_prompt: str | None = None,
    ) -> GenerationResult:
        confidence_result = compute_confidence(ranked_chunks)
        confidence_gate = gate(confidence_result.score)
        active_system_prompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT
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
                confidence_score=confidence_result.score,
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
                confidence_score=confidence_result.score,
                latency_ms=latency_ms,
            )

        confidence_level = "high" if confidence_gate == ConfidenceGate.HIGH else "hedged"
        messages = build_prompt(query, ranked_chunks, confidence_level=confidence_level)

        # Prepend last 5 turns (10 messages) of conversation history
        if conversation_history:
            messages = conversation_history[-10:] + messages

        start = time.monotonic()
        response = self._llm.complete(active_system_prompt, messages, max_tokens=MAX_ANSWER_TOKENS)
        latency_ms = (time.monotonic() - start) * 1000

        answer, hallucinated = self._check_citations(response.content, len(ranked_chunks))
        if hallucinated:
            answer = (
                "WARNING: The following answer contained hallucinated citations "
                f"([Doc {', '.join(str(n) for n in hallucinated)}]) that were removed.\n\n"
                + answer
            )

        citations = sorted(
            {int(m) for m in _CITATION_RE.findall(answer)} - set(hallucinated)
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
            confidence_score=confidence_result.score,
            hallucinated_citations=hallucinated,
            latency_ms=latency_ms,
        )

    @staticmethod
    def _check_citations(answer: str, num_docs: int) -> tuple[str, list[int]]:
        """Strip citations referencing non-existent docs; return (cleaned_answer, hallucinated_list)."""
        hallucinated: list[int] = []
        for m in _CITATION_RE.finditer(answer):
            n = int(m.group(1))
            if n < 1 or n > num_docs:
                if n not in hallucinated:
                    hallucinated.append(n)

        if not hallucinated:
            return answer, []

        cleaned = answer
        for n in sorted(hallucinated):
            cleaned = cleaned.replace(f"[Doc {n}]", "")
        return cleaned, sorted(hallucinated)
