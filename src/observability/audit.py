"""
Immutable audit log — every query and answer persisted to Postgres/SQLite.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from src.db.models import AuditLog, Base
from src.generation.confidence import gate as compute_gate

if TYPE_CHECKING:
    from src.generation.card_generator import TreatmentCardResult
    from src.generation.confidence import ConfidenceGate
    from src.generation.generator import GenerationResult
    from src.retrieval.retriever import RetrievalResult


class AuditLogger:
    """INSERT-only audit log writer backed by SQLAlchemy (sync engine)."""

    def __init__(self, db_url: str) -> None:
        # asyncpg is an async-only driver; swap to psycopg2 for sync create_engine
        sync_url = db_url.replace("postgresql+asyncpg://", "postgresql://")
        connect_args = {"check_same_thread": False} if sync_url.startswith("sqlite") else {}
        self._engine = create_engine(sync_url, connect_args=connect_args)
        Base.metadata.create_all(self._engine)

    @property
    def engine(self):
        return self._engine

    async def log(
        self,
        query_id: str,
        question: str,
        result: "GenerationResult",
        retrieval_result: "RetrievalResult",
        confidence: float,
        gate: "ConfidenceGate",
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        """Persist one audit record. Non-blocking via asyncio.to_thread."""
        sources = [
            {
                "pmid": c.metadata.get("pmid"),
                "title": c.metadata.get("title"),
                "section": c.metadata.get("section"),
                "score": round(c.relevance_score, 4),
            }
            for c in retrieval_result.chunks
        ]
        total_latency = sum(retrieval_result.latency_ms.values()) + result.latency_ms

        await asyncio.to_thread(
            self._insert,
            query_id=query_id,
            question=question,
            answer=result.answer,
            confidence=confidence,
            gate_decision=gate.value,
            model=result.model_used,
            provider=result.provider,
            input_tokens=result.prompt_tokens,
            output_tokens=result.completion_tokens,
            latency_ms=total_latency,
            sources=sources,
            user_id=user_id,
            session_id=session_id,
            hallucinated_citations=result.hallucinated_citations,
            flagged=bool(result.hallucinated_citations),
        )

    def _insert(self, **kwargs) -> None:
        """Synchronous INSERT — called via asyncio.to_thread."""
        with Session(self._engine) as session:
            row = AuditLog(
                timestamp=datetime.now(timezone.utc),
                **kwargs,
            )
            session.add(row)
            session.commit()

    async def log_treatment_card(
        self,
        query_id: str,
        patient_id: str,
        clinical_history: str,
        card_result: "TreatmentCardResult",
        model: str = "",
        provider: str = "",
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        """Persist one /treatment-card record. Non-blocking via asyncio.to_thread.

        Reuses the same audit_log table as /query: `question` holds a short
        patient/clinical-history summary, `answer` holds the full structured
        card (including chunks_used/grounded — independent of whether the
        caller opted into disclose_fallback at the API level) as JSON.

        `sources` always holds the real, database-grounded sources_detail —
        even when it's an empty list (zero chunks retrieved). It is never
        substituted with the LLM's free-text `sources`, so a reviewer can't
        mistake an ungrounded card for a grounded one just by looking at the
        shape of this column; the LLM's own (possibly hallucinated) `sources`
        text is preserved separately inside `answer`.
        """
        retrieval_meta = card_result.retrieval_metadata or {}
        sources_detail = retrieval_meta.get("sources_detail") or []
        chunks_used = retrieval_meta.get("chunks_used", 0)
        confidence = retrieval_meta.get("confidence_score", 0.0)
        grounded = retrieval_meta.get("grounded", chunks_used > 0)

        card_json = json.dumps({
            "stage": card_result.stage,
            "confidence": card_result.confidence,
            "guideline": card_result.guideline,
            "comorbidities_impact": card_result.comorbidities_impact,
            "treatment": [asdict(t) for t in card_result.treatment],
            "treatment_confidence": card_result.treatment_confidence,
            "sources": card_result.sources,
            "retrieval_metadata": {
                "chunks_used": chunks_used,
                "grounded": grounded,
                "confidence_score": confidence,
            },
        }, ensure_ascii=False)

        await asyncio.to_thread(
            self._insert,
            query_id=query_id,
            question=f"[treatment-card] patient={patient_id}: {clinical_history[:500]}",
            answer=card_json,
            confidence=confidence,
            gate_decision=compute_gate(confidence).value,
            model=model,
            provider=provider,
            input_tokens=card_result.prompt_tokens,
            output_tokens=card_result.completion_tokens,
            latency_ms=card_result.latency_ms,
            sources=sources_detail,
            user_id=user_id,
            session_id=session_id,
            hallucinated_citations=[],
            flagged=not grounded,
        )
