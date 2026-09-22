"""
POST /treatment-card — structured treatment card generation endpoint.

Default output language is French (matches the original, long-standing
behaviour relied on by existing callers); pass language="en" for English.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import TYPE_CHECKING, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from src.api.middleware.auth import api_key_fingerprint, require_api_key
from src.api.model_selection import card_generator_for
from src.generation.models import ModelNotAllowed
from src.observability.logging import get_logger, query_id_var
from config.constants import CONFIDENCE_REFUSE, normalise_topic

if TYPE_CHECKING:
    from src.generation.card_generator import CardGenerator
    from src.observability.audit import AuditLogger
    from src.retrieval.retriever import RAGRetriever

router = APIRouter(tags=["treatment-card"])
logger = get_logger(__name__)


# ── Request / Response schemas ────────────────────────────────────────────────

class TreatmentCardRequest(BaseModel):
    patient_id: str = Field(..., min_length=1, max_length=100)
    cancer_type: str = Field(..., min_length=1, max_length=100)
    age_range: str = Field(default="")
    clinical_history: str = Field(..., min_length=10, max_length=10000)
    comorbidities: dict[str, str] = Field(default_factory=dict)
    top_k: int = Field(default=5, ge=1, le=10)
    system_prompt: str | None = Field(default=None, max_length=10000)
    conversation_id: str | None = None
    model: str | None = Field(
        default=None,
        description=(
            "Generation model for this request. Must be one of the allowlisted "
            "ids in src.generation.models; anything else is rejected with 400. "
            "None means use the server default (GENERATION_MODEL)."
        ),
    )
    language: Literal["fr", "en"] = Field(
        default="fr",
        description=(
            "Language for server-injected text (prompt scaffolding, fallback "
            "values, warning text). Does not override an explicit system_prompt's "
            "own instructions. Default 'fr' preserves pre-existing behaviour."
        ),
    )
    keep_citations: bool = Field(
        default=True,
        description=(
            "If true, treatment[].drug may carry a range-validated [Doc N] tag, "
            "and `sources` is regenerated from real chunk metadata for every "
            "[Doc N] the model referenced — hallucination-free by construction. "
            "Default true. Set to false only to replicate pre-existing behaviour "
            "(LLM writes free-text sources, which may hallucinate author/journal)."
        ),
    )
    disclose_fallback: bool = Field(
        default=True,
        description=(
            "When no chunks were retrieved, the card is generated from the "
            "model's own clinical knowledge rather than from the indexed "
            "literature. That is intended behaviour, but such a card looks "
            "identical to a grounded one. If true (the default), `sources` and "
            "`sources_detail` are replaced with an explicit disclosure instead "
            "of being left empty. Set to false only to reproduce the older, "
            "quieter behaviour. Note that retrieval_metadata['grounded'] is "
            "reported either way and does not depend on this flag."
        ),
    )

    @field_validator("comorbidities", mode="before")
    @classmethod
    def coerce_none_dict(cls, v: Any) -> dict:
        return v or {}

    @field_validator("cancer_type")
    @classmethod
    def normalise_cancer_type(cls, v: str) -> str:
        return normalise_topic(v)


class TreatmentWarningOut(BaseModel):
    type: str
    drug: str
    jurisdiction: str
    message: str


class TreatmentTripletOut(BaseModel):
    drug: str
    intent: str
    level: str
    warnings: list[TreatmentWarningOut]


class TreatmentCardResponse(BaseModel):
    patient_id: str
    stage: str
    confidence: str
    guideline: str
    comorbidities_impact: str
    treatment: list[TreatmentTripletOut]
    treatment_confidence: str
    sources: list[str]
    retrieval_metadata: dict
    request_id: str
    latency_ms: int
    model_used: str = Field(
        default="",
        description="The generation model that actually produced this card.",
    )


# ── Dependency accessors ──────────────────────────────────────────────────────

def get_retriever(request: Request) -> "RAGRetriever | None":
    return getattr(request.app.state, "retriever", None)


def get_card_generator(request: Request) -> "CardGenerator | None":
    return getattr(request.app.state, "card_generator", None)


def get_audit_logger(request: Request) -> "AuditLogger | None":
    return getattr(request.app.state, "audit_logger", None)


# ── Route handler ─────────────────────────────────────────────────────────────

@router.post("/treatment-card", response_model=TreatmentCardResponse)
async def treatment_card_endpoint(
    body: TreatmentCardRequest,
    request: Request,
    retriever: "RAGRetriever | None" = Depends(get_retriever),
    card_generator: "CardGenerator | None" = Depends(get_card_generator),
    audit_logger: "AuditLogger | None" = Depends(get_audit_logger),
    _api_key: str = Depends(require_api_key),
) -> Any:
    if retriever is None or card_generator is None:
        raise HTTPException(status_code=503, detail="Service not initialised")

    try:
        card_generator = card_generator_for(request, body.model, card_generator)
    except ModelNotAllowed as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None

    query_id = str(uuid.uuid4())
    query_id_var.set(query_id)
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    t_total = time.perf_counter()

    # ── Step 1: translate clinical narrative to English for retrieval ──────
    narrative = _build_narrative(body)
    try:
        english_query = await asyncio.to_thread(
            card_generator.translate_to_english, narrative
        )
    except Exception as exc:
        logger.warning("Translation failed, falling back to raw narrative: %s", exc)
        english_query = narrative

    # ── Step 2: retrieve relevant chunks ──────────────────────────────────
    filters: dict = {"cancer_type": [body.cancer_type]}
    try:
        retrieval_result = await asyncio.to_thread(
            retriever.retrieve,
            english_query,
            filters=filters,
            top_k_rerank=body.top_k,
        )
    except Exception as exc:
        logger.error("Retrieval failed: %s", exc)
        raise HTTPException(status_code=503, detail="Retrieval service unavailable")

    # ── Step 3: generate card ─────────────────────────────────────────────
    # Low-confidence gate: if retrieval is poor, pass no chunks so the LLM
    # generates from parametric knowledge and marks confidence as low.
    # The retrieval_metadata still reflects the actual retrieval score.
    if retrieval_result.retrieval_confidence < CONFIDENCE_REFUSE:
        logger.warning(
            "Low retrieval confidence (%.2f) for patient %s — falling back to parametric knowledge",
            retrieval_result.retrieval_confidence,
            body.patient_id,
        )
        chunks_for_generation = []
    else:
        chunks_for_generation = retrieval_result.chunks

    try:
        card_result = await asyncio.to_thread(
            card_generator.generate_card,
            patient_id=body.patient_id,
            cancer_type=body.cancer_type,
            age_range=body.age_range,
            clinical_history=body.clinical_history,
            comorbidities=body.comorbidities,
            ranked_chunks=chunks_for_generation,
            confidence_score=retrieval_result.retrieval_confidence,
            system_prompt=body.system_prompt,
            language=body.language,
            keep_citations=body.keep_citations,
            disclose_fallback=body.disclose_fallback,
        )
    except Exception as exc:
        logger.error("Card generation failed: %s", exc)
        raise HTTPException(status_code=503, detail="Generation service unavailable")

    total_ms = int((time.perf_counter() - t_total) * 1000)

    # ── Audit log (fire-and-forget, never raise) ──────────────────────────
    if audit_logger is not None:
        try:
            await audit_logger.log_treatment_card(
                query_id=query_id,
                patient_id=body.patient_id,
                clinical_history=body.clinical_history,
                card_result=card_result,
                model=getattr(card_generator, "model", ""),
                provider=getattr(card_generator, "provider", ""),
                user_id=api_key_fingerprint(_api_key),
                session_id=body.conversation_id,
            )
        except Exception as exc:
            logger.warning("Audit log failed: %s", exc)

    return TreatmentCardResponse(
        patient_id=card_result.patient_id,
        stage=card_result.stage,
        confidence=card_result.confidence,
        guideline=card_result.guideline,
        comorbidities_impact=card_result.comorbidities_impact,
        treatment=[
            TreatmentTripletOut(
                drug=t.drug,
                intent=t.intent,
                level=t.level,
                warnings=[
                    TreatmentWarningOut(
                        type=w.type,
                        drug=w.drug,
                        jurisdiction=w.jurisdiction,
                        message=w.message,
                    )
                    for w in t.warnings
                ],
            )
            for t in card_result.treatment
        ],
        treatment_confidence=card_result.treatment_confidence,
        sources=card_result.sources,
        retrieval_metadata=card_result.retrieval_metadata,
        request_id=request_id,
        latency_ms=total_ms,
        model_used=getattr(card_generator, "model", ""),
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_narrative(body: TreatmentCardRequest) -> str:
    """Assemble the clinical narrative to pass to the translation step."""
    parts = [f"Cancer type: {body.cancer_type}"]
    if body.age_range:
        parts.append(f"Age: {body.age_range}")
    parts.append(body.clinical_history)
    if body.comorbidities:
        comorb = ", ".join(f"{k}: {v}" for k, v in body.comorbidities.items())
        parts.append(f"Comorbidities: {comorb}")
    return "\n".join(parts)
