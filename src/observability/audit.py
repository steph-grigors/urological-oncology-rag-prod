"""
Immutable audit log — every query and answer persisted to Postgres/SQLite.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from src.db.models import AuditLog, Base

if TYPE_CHECKING:
    from src.generation.confidence import ConfidenceGate
    from src.generation.generator import GenerationResult
    from src.retrieval.retriever import RetrievalResult


class AuditLogger:
    """INSERT-only audit log writer backed by SQLAlchemy (sync engine)."""

    def __init__(self, db_url: str) -> None:
        # Use check_same_thread=False for SQLite in async contexts
        connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}
        self._engine = create_engine(db_url, connect_args=connect_args)
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
