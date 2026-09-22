"""
POST /query — main RAG query endpoint.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from config.constants import normalise_topic
from src.api.middleware.auth import api_key_fingerprint, require_api_key
from src.api.model_selection import generator_for
from src.generation.models import ModelNotAllowed
from src.generation.confidence import gate
from src.generation.source_card import chunk_to_source_detail
from src.observability.logging import get_logger, query_id_var

if TYPE_CHECKING:
    from src.db.document_store import DocumentStore
    from src.generation.generator import ClinicalGenerator
    from src.observability.audit import AuditLogger
    from src.retrieval.retriever import RAGRetriever

router = APIRouter(tags=["query"])
logger = get_logger(__name__)


# ── Request / Response schemas ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=10000)
    cancer_types: list[str] = Field(default_factory=list)
    year_from: int | None = None
    year_to: int | None = None
    study_designs: list[str] = Field(default_factory=list)
    conversation_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=10)
    stream: bool = False
    system_prompt: str | None = Field(default=None, max_length=10000)
    model: str | None = Field(
        default=None,
        description=(
            "Generation model for this request. Must be one of the allowlisted "
            "ids in src.generation.models; anything else is rejected with 400. "
            "None means use the server default (GENERATION_MODEL)."
        ),
    )

    @field_validator("cancer_types", mode="before")
    @classmethod
    def coerce_none_list(cls, v: Any) -> list:
        return v or []

    @field_validator("cancer_types")
    @classmethod
    def normalise_cancer_types(cls, v: list[str]) -> list[str]:
        return [normalise_topic(raw) for raw in v]

    @field_validator("study_designs", mode="before")
    @classmethod
    def coerce_none_list2(cls, v: Any) -> list:
        return v or []


class SourceCard(BaseModel):
    chunk_id: str
    title: str
    authors: str
    journal: str
    year: int | None
    study_design: str
    sample_size: int | None
    section: str
    key_finding: str
    pmid: str


class LatencyBreakdown(BaseModel):
    retrieval: int
    rerank: int
    generation: int
    total: int


class QueryQualityScores(BaseModel):
    """Structural checks on this answer, computed without an LLM call.

    These are lexical and structural measures. None of them reads the medical
    content of the answer or judges whether it is clinically correct, and they
    must not be presented as if they did. They are cheap guards against
    specific, mechanical failure modes -- a citation pointing at a document
    that was never retrieved, an answer that ignores the question's subject,
    a retrieval set that missed the topic entirely.

    Field names are kept as-is for wire compatibility with existing clients;
    the descriptions below say what each one actually measures. See
    src/evaluation/judges.py for the implementations.
    """

    faithfulness: float = Field(
        ...,
        description=(
            "Citation validity. The fraction of [Doc N] tags in the answer "
            "whose N points at a chunk that was actually retrieved. It does "
            "NOT check that the cited chunk supports the claim. Note that "
            "ClinicalGenerator strips out-of-range tags before this runs, so "
            "in practice this is 1.0 whenever the answer cites anything, and "
            "0.85 when it cites nothing at all."
        ),
    )
    answer_relevance: float = Field(
        ...,
        description=(
            "Query-term coverage. The fraction of the question's content words "
            "that appear anywhere in the answer. Rewards restating the "
            "question and says nothing about whether the answer is right."
        ),
    )
    context_precision: float = Field(
        ...,
        description=(
            "Retrieved-chunk term overlap. The fraction of retrieved chunks "
            "containing at least one of the question's content words. A "
            "topicality check on retrieval, not a relevance judgement."
        ),
    )
    method: str = Field(
        default="lexical_heuristic",
        description=(
            "How these scores were produced. 'lexical_heuristic' means string "
            "and regex matching with no model call and no clinical judgement. "
            "Clients should label them accordingly."
        ),
    )


class QueryResponse(BaseModel):
    answer: str
    evidence_quality: str = Field(
        ...,
        description=(
            "Response posture chosen from confidence_score: 'high', 'hedged', "
            "'caveated', or 'insufficient' when nothing relevant was retrieved "
            "and the answer comes from the model's own knowledge."
        ),
    )
    confidence_score: float = Field(
        ...,
        description=(
            "How well retrieval matched the question, NOT confidence that the "
            "answer is correct. It is the mean reranker relevance of the top "
            "candidates, adjusted for evidence level, source diversity and "
            "score spread. A high value means the retrieved passages look like "
            "the question; it does not mean they answer it. A question with no "
            "literature can still score highly if the corpus contains passages "
            "on the same subject. Nothing here reads the passages to judge "
            "whether they address what was asked."
        ),
    )
    sources: list[SourceCard]
    conversation_id: str
    request_id: str
    latency_ms: LatencyBreakdown
    quality: QueryQualityScores | None = None
    model_used: str = Field(
        default="",
        description="The generation model that actually answered this request.",
    )


# ── Dependency accessors ──────────────────────────────────────────────────────

def get_retriever(request: Request) -> "RAGRetriever | None":
    return getattr(request.app.state, "retriever", None)


def get_generator(request: Request) -> "ClinicalGenerator | None":
    return getattr(request.app.state, "generator", None)


def get_audit_logger(request: Request) -> "AuditLogger | None":
    return getattr(request.app.state, "audit_logger", None)


def get_document_store(request: Request) -> "DocumentStore | None":
    return getattr(request.app.state, "document_store", None)


# ── Route handler ─────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
async def query_endpoint(
    body: QueryRequest,
    request: Request,
    retriever: "RAGRetriever | None" = Depends(get_retriever),
    generator: "ClinicalGenerator | None" = Depends(get_generator),
    audit_logger: "AuditLogger | None" = Depends(get_audit_logger),
    document_store: "DocumentStore | None" = Depends(get_document_store),
    _api_key: str = Depends(require_api_key),
) -> Any:
    if retriever is None or generator is None:
        raise HTTPException(status_code=503, detail="Service not initialised")

    try:
        generator = generator_for(request, body.model, generator)
    except ModelNotAllowed as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None

    query_id = str(uuid.uuid4())
    query_id_var.set(query_id)
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    t_total = time.perf_counter()

    # Build filters dict from request fields
    filters: dict = {}
    if body.cancer_types:
        filters["cancer_type"] = body.cancer_types
    if body.study_designs:
        filters["study_design"] = body.study_designs
    if body.year_from is not None:
        filters["year_min"] = body.year_from
    if body.year_to is not None:
        filters["year_max"] = body.year_to

    # ── Retrieval ─────────────────────────────────────────────────────────
    try:
        retrieval_result = await asyncio.to_thread(
            retriever.retrieve,
            body.query,
            filters=filters or None,
            top_k_rerank=body.top_k,
        )
    except Exception as exc:
        logger.error("Retrieval failed: %s", exc)
        raise HTTPException(status_code=503, detail="Retrieval service unavailable")

    # `retrieval_result.latency_ms` holds overlapping spans: "total_ms" already
    # covers embed + dense + bm25 + rerank + any web fallback, so summing every
    # value double-counted each phase and then added the total on top again.
    # Report the two phases the client actually distinguishes: everything the
    # retriever did except reranking, and reranking on its own.
    timings = retrieval_result.latency_ms
    rerank_ms = int(timings.get("rerank_ms", 0))
    retr_ms = max(0, int(timings.get("total_ms", 0)) - rerank_ms)

    # ── Conversation history fetch ─────────────────────────────────────────
    conversation_history: list[dict] | None = None
    if body.conversation_id and document_store is not None:
        try:
            conversation_history = await document_store.get_conversation_history(
                body.conversation_id, limit=10
            )
        except Exception as exc:
            logger.warning("Failed to fetch conversation history: %s", exc)

    # ── Generation ────────────────────────────────────────────────────────
    t_gen = time.perf_counter()
    try:
        gen_result = await asyncio.to_thread(
            generator.generate,
            body.query,
            retrieval_result.chunks,
            conversation_history=conversation_history,
            system_prompt=body.system_prompt,
            # Scored over the full reranked candidate set. Letting the
            # generator recompute would see only the chunks that survived
            # grading, which cannot fall below the grading threshold.
            confidence_score=retrieval_result.retrieval_confidence,
        )
    except Exception as exc:
        logger.error("Generation failed: %s", exc)
        raise HTTPException(status_code=503, detail="Generation service unavailable")
    gen_ms = int((time.perf_counter() - t_gen) * 1000)

    total_ms = int((time.perf_counter() - t_total) * 1000)
    confidence_gate = gate(gen_result.confidence_score)

    # ── Audit log (fire-and-forget, never raise) ──────────────────────────
    if audit_logger is not None:
        try:
            await audit_logger.log(
                query_id=query_id,
                question=body.query,
                result=gen_result,
                retrieval_result=retrieval_result,
                confidence=gen_result.confidence_score,
                gate=confidence_gate,
                user_id=api_key_fingerprint(_api_key),
                session_id=body.conversation_id,
            )
        except Exception as exc:
            logger.warning("Audit log failed: %s", exc)

    # ── Persist conversation turn (fire-and-forget, never raise) ─────────
    conversation_id = body.conversation_id or query_id
    if body.conversation_id and document_store is not None:
        try:
            await document_store.append_conversation_turns(
                body.conversation_id, body.query, gen_result.answer
            )
        except Exception as exc:
            logger.warning("Failed to persist conversation turn: %s", exc)

    # ── Build source cards ────────────────────────────────────────────────
    sources = [_to_source_card(c) for c in retrieval_result.chunks]

    # ── Structural checks (lexical only, no LLM call -- safe inline) ──────
    # Not a measure of clinical accuracy; see QueryQualityScores.
    quality: QueryQualityScores | None = None
    try:
        from src.evaluation.judges import JudgeSet

        judge_scores = JudgeSet().score_all(
            question=body.query,
            answer=gen_result.answer,
            chunks=retrieval_result.chunks,
        )
        quality = QueryQualityScores(
            faithfulness=round(judge_scores.faithfulness, 4),
            answer_relevance=round(judge_scores.answer_relevance, 4),
            context_precision=round(judge_scores.context_precision, 4),
        )
    except Exception as exc:
        logger.warning("Quality scoring failed: %s", exc)

    response_body = QueryResponse(
        answer=gen_result.answer,
        evidence_quality=gen_result.evidence_quality,
        confidence_score=round(gen_result.confidence_score, 4),
        sources=sources,
        conversation_id=conversation_id,
        request_id=request_id,
        latency_ms=LatencyBreakdown(
            retrieval=retr_ms,
            rerank=rerank_ms,
            generation=gen_ms,
            total=total_ms,
        ),
        quality=quality,
        model_used=gen_result.model_used or getattr(generator, "model", ""),
    )

    if body.stream:
        return _sse_response(response_body)

    return response_body


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_source_card(chunk) -> SourceCard:
    return SourceCard(**asdict(chunk_to_source_detail(chunk)))


def _sse_response(body: QueryResponse) -> StreamingResponse:
    payload = json.dumps(body.model_dump())

    async def _generate():
        yield f"data: {payload}\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")
