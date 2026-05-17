"""
POST /treatment-card — structured French treatment card generation endpoint.
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from src.api.middleware.auth import require_api_key
from src.observability.logging import get_logger, query_id_var

if TYPE_CHECKING:
    from src.generation.card_generator import CardGenerator
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

    @field_validator("comorbidities", mode="before")
    @classmethod
    def coerce_none_dict(cls, v: Any) -> dict:
        return v or {}


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


# ── Dependency accessors ──────────────────────────────────────────────────────

def get_retriever(request: Request) -> "RAGRetriever | None":
    return getattr(request.app.state, "retriever", None)


def get_card_generator(request: Request) -> "CardGenerator | None":
    return getattr(request.app.state, "card_generator", None)


# ── Route handler ─────────────────────────────────────────────────────────────

@router.post("/treatment-card", response_model=TreatmentCardResponse)
async def treatment_card_endpoint(
    body: TreatmentCardRequest,
    request: Request,
    retriever: "RAGRetriever | None" = Depends(get_retriever),
    card_generator: "CardGenerator | None" = Depends(get_card_generator),
    _api_key: str = Depends(require_api_key),
) -> Any:
    if retriever is None or card_generator is None:
        raise HTTPException(status_code=503, detail="Service not initialised")

    query_id = str(uuid.uuid4())
    query_id_var.set(query_id)
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    t_total = time.perf_counter()

    # ── Step 1: translate clinical narrative to English for retrieval ──────
    narrative = _build_narrative(body)
    try:
        english_query = card_generator.translate_to_english(narrative)
    except Exception as exc:
        logger.warning("Translation failed, falling back to raw narrative: %s", exc)
        english_query = narrative

    # ── Step 2: retrieve relevant chunks ──────────────────────────────────
    filters: dict = {"cancer_type": [body.cancer_type]}
    try:
        retrieval_result = retriever.retrieve(
            english_query,
            filters=filters,
            top_k_rerank=body.top_k,
        )
    except Exception as exc:
        logger.error("Retrieval failed: %s", exc)
        raise HTTPException(status_code=503, detail="Retrieval service unavailable")

    # ── Step 3: generate card ─────────────────────────────────────────────
    try:
        card_result = card_generator.generate_card(
            patient_id=body.patient_id,
            cancer_type=body.cancer_type,
            age_range=body.age_range,
            clinical_history=body.clinical_history,
            comorbidities=body.comorbidities,
            ranked_chunks=retrieval_result.chunks,
            confidence_score=retrieval_result.retrieval_confidence,
            system_prompt=body.system_prompt,
        )
    except Exception as exc:
        logger.error("Card generation failed: %s", exc)
        raise HTTPException(status_code=503, detail="Generation service unavailable")

    total_ms = int((time.perf_counter() - t_total) * 1000)

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
