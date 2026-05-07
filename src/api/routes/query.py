"""
POST /query — main RAG query endpoint.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from src.api.middleware.auth import require_api_key
from src.generation.confidence import gate
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

    @field_validator("cancer_types", mode="before")
    @classmethod
    def coerce_none_list(cls, v: Any) -> list:
        return v or []

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


class QueryResponse(BaseModel):
    answer: str
    evidence_quality: str
    confidence_score: float
    sources: list[SourceCard]
    conversation_id: str
    request_id: str
    latency_ms: LatencyBreakdown


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
        retrieval_result = retriever.retrieve(
            body.query,
            filters=filters or None,
            top_k_rerank=body.top_k,
        )
    except Exception as exc:
        logger.error("Retrieval failed: %s", exc)
        raise HTTPException(status_code=503, detail="Retrieval service unavailable")

    retr_ms = int(sum(retrieval_result.latency_ms.values()))
    rerank_ms = int(retrieval_result.latency_ms.get("rerank_ms", 0))

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
        gen_result = generator.generate(
            body.query, retrieval_result.chunks, conversation_history=conversation_history
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
                user_id=_api_key if _api_key not in ("", "dev") else None,
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
    )

    if body.stream:
        return _sse_response(response_body)

    return response_body


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_source_card(chunk) -> SourceCard:
    meta = chunk.metadata if hasattr(chunk, "metadata") else {}
    authors_raw = meta.get("authors", [])
    if isinstance(authors_raw, list):
        authors_str = ", ".join(str(a) for a in authors_raw[:3])
        if len(authors_raw) > 3:
            authors_str += " et al."
    else:
        authors_str = str(authors_raw) if authors_raw else ""

    text = chunk.text if hasattr(chunk, "text") else ""
    return SourceCard(
        chunk_id=chunk.chunk_id if hasattr(chunk, "chunk_id") else "",
        title=meta.get("title") or "Unknown",
        authors=authors_str,
        journal=meta.get("journal") or "",
        year=meta.get("year"),
        study_design=meta.get("study_design") or "",
        sample_size=meta.get("sample_size"),
        section=meta.get("section") or "",
        key_finding=text[:150],
        pmid=meta.get("pmid") or "",
    )


def _sse_response(body: QueryResponse) -> StreamingResponse:
    payload = json.dumps(body.model_dump())

    async def _generate():
        yield f"data: {payload}\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")
