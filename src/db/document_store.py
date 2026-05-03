"""
Postgres document store abstraction.

Postgres serves three roles in this system:
    1. Full-text search index (tsvector) for BM25-style keyword retrieval.
    2. Persistent store for the audit log (`audit_log` table).
    3. Metadata store for papers and chunks.

Uses a synchronous SQLAlchemy engine wrapped in asyncio.to_thread for
async callers — same pattern as AuditLogger (asyncpg is not available).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from src.db.models import AuditLog, Base, Chunk as ChunkModel, ConversationHistory, Paper

logger = logging.getLogger(__name__)


# ── Return types ──────────────────────────────────────────────────────────────

@dataclass
class BM25Result:
    chunk_id: str
    text: str
    rank: float
    pmc_id: str
    title: str
    section: str
    metadata: dict = field(default_factory=dict)


@dataclass
class SearchFilter:
    cancer_types: list[str] = field(default_factory=list)
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    study_designs: list[str] = field(default_factory=list)


@dataclass
class CorpusStats:
    total_chunks: int = 0
    total_papers: int = 0
    by_topic: dict[str, int] = field(default_factory=dict)
    by_year: dict[int, int] = field(default_factory=dict)
    by_study_design: dict[str, int] = field(default_factory=dict)


@dataclass
class AuditRecord:
    query_id: str
    question: str
    answer: str
    confidence: float
    gate_decision: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    sources: list
    hallucinated_citations: list = field(default_factory=list)
    rewritten_query: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    flagged: bool = False


# ── DocumentStore ─────────────────────────────────────────────────────────────

class DocumentStore:
    def __init__(self, db_url: str) -> None:
        # asyncpg is an async-only driver; swap to psycopg2 for sync create_engine
        sync_url = db_url.replace("postgresql+asyncpg://", "postgresql://")
        connect_args = {"check_same_thread": False} if sync_url.startswith("sqlite") else {}
        self._engine = create_engine(sync_url, connect_args=connect_args)
        Base.metadata.create_all(self._engine)
        self._is_postgres = "postgresql" in sync_url or "postgres" in sync_url

    # ── Public async API ──────────────────────────────────────────────────────

    async def upsert_chunks(self, chunks: list) -> None:
        """Insert or update chunk rows (and their parent papers) in Postgres."""
        await asyncio.to_thread(self._upsert_chunks_sync, chunks)

    async def full_text_search(
        self,
        query: str,
        top_k: int = 10,
        filter: SearchFilter | None = None,
    ) -> list[BM25Result]:
        return await asyncio.to_thread(self._full_text_search_sync, query, top_k, filter)

    async def get_chunk(self, chunk_id: str):
        return await asyncio.to_thread(self._get_chunk_sync, chunk_id)

    async def write_audit(self, record: AuditRecord) -> None:
        await asyncio.to_thread(self._write_audit_sync, record)

    async def get_corpus_stats(self) -> CorpusStats:
        return await asyncio.to_thread(self._get_corpus_stats_sync)

    async def get_conversation_history(
        self, conversation_id: str, limit: int = 10
    ) -> list[dict]:
        """Return the last `limit` turns as [{"role": ..., "content": ...}, ...] ordered oldest-first."""
        return await asyncio.to_thread(
            self._get_conversation_history_sync, conversation_id, limit
        )

    async def append_conversation_turns(
        self, conversation_id: str, question: str, answer: str
    ) -> None:
        """Persist one user + one assistant turn for the given conversation."""
        await asyncio.to_thread(
            self._append_conversation_turns_sync, conversation_id, question, answer
        )

    # ── Synchronous implementations ───────────────────────────────────────────

    def _upsert_chunks_sync(self, chunks: list) -> None:
        with Session(self._engine) as session:
            for chunk in chunks:
                meta = chunk.metadata
                # Upsert paper row first
                paper = session.get(Paper, meta.pmcid)
                if paper is None:
                    paper = Paper(
                        pmc_id=meta.pmcid,
                        pmid=meta.pmid,
                        title=meta.title,
                        authors=meta.authors,
                        journal=meta.journal,
                        year=meta.year,
                        study_design=meta.study_design,
                        intervention=getattr(meta, "intervention", None),
                        comparator=getattr(meta, "comparator", None),
                    )
                    session.add(paper)

                # Upsert chunk row
                orm_chunk = session.get(ChunkModel, chunk.id)
                if orm_chunk is None:
                    orm_chunk = ChunkModel(
                        id=chunk.id,
                        pmc_id=meta.pmcid,
                        text=chunk.text,
                        section_name=meta.section,
                        section_type=meta.chunk_type,
                        chunk_index=meta.chunk_index,
                    )
                    session.add(orm_chunk)
                else:
                    orm_chunk.text = chunk.text
                    orm_chunk.section_name = meta.section

            session.commit()
        logger.debug("upsert_chunks: wrote %d chunks", len(chunks))

    def _full_text_search_sync(
        self, query: str, top_k: int, filter: SearchFilter | None
    ) -> list[BM25Result]:
        with Session(self._engine) as session:
            if self._is_postgres:
                return self._postgres_fts(session, query, top_k, filter)
            return self._sqlite_fts_fallback(session, query, top_k, filter)

    def _postgres_fts(
        self, session: Session, query: str, top_k: int, filter: SearchFilter | None
    ) -> list[BM25Result]:
        sql = text("""
            SELECT
                c.id          AS chunk_id,
                c.text        AS text,
                c.section_name AS section,
                p.pmc_id,
                p.title,
                ts_rank_cd(c.tsvector_col, plainto_tsquery('english', :q)) AS rank
            FROM chunks c
            JOIN papers p ON c.pmc_id = p.pmc_id
            WHERE c.tsvector_col @@ plainto_tsquery('english', :q)
            ORDER BY rank DESC
            LIMIT :top_k
        """)
        rows = session.execute(sql, {"q": query, "top_k": top_k}).fetchall()
        return [
            BM25Result(
                chunk_id=r.chunk_id,
                text=r.text,
                rank=float(r.rank),
                pmc_id=r.pmc_id,
                title=r.title or "",
                section=r.section or "",
            )
            for r in rows
        ]

    def _sqlite_fts_fallback(
        self, session: Session, query: str, top_k: int, filter: SearchFilter | None
    ) -> list[BM25Result]:
        terms = query.split()
        if not terms:
            return []
        like_clause = " AND ".join(f"c.text LIKE :term{i}" for i in range(len(terms)))
        sql = text(f"""
            SELECT c.id AS chunk_id, c.text AS text, c.section_name AS section,
                   p.pmc_id, p.title
            FROM chunks c JOIN papers p ON c.pmc_id = p.pmc_id
            WHERE {like_clause}
            LIMIT :top_k
        """)
        params: dict = {f"term{i}": f"%{t}%" for i, t in enumerate(terms)}
        params["top_k"] = top_k
        rows = session.execute(sql, params).fetchall()
        return [
            BM25Result(
                chunk_id=r.chunk_id,
                text=r.text,
                rank=1.0,
                pmc_id=r.pmc_id,
                title=r.title or "",
                section=r.section or "",
            )
            for r in rows
        ]

    def _get_chunk_sync(self, chunk_id: str):
        with Session(self._engine) as session:
            return session.get(ChunkModel, chunk_id)

    def _write_audit_sync(self, record: AuditRecord) -> None:
        with Session(self._engine) as session:
            session.add(AuditLog(
                query_id=record.query_id,
                timestamp=datetime.now(timezone.utc),
                question=record.question,
                rewritten_query=record.rewritten_query,
                answer=record.answer,
                confidence=record.confidence,
                gate_decision=record.gate_decision,
                model=record.model,
                provider=record.provider,
                input_tokens=record.input_tokens,
                output_tokens=record.output_tokens,
                latency_ms=record.latency_ms,
                sources=record.sources,
                user_id=record.user_id,
                session_id=record.session_id,
                hallucinated_citations=record.hallucinated_citations,
                flagged=record.flagged,
            ))
            session.commit()

    def _get_conversation_history_sync(
        self, conversation_id: str, limit: int
    ) -> list[dict]:
        with Session(self._engine) as session:
            rows = (
                session.query(ConversationHistory)
                .filter(ConversationHistory.conversation_id == conversation_id)
                .order_by(ConversationHistory.created_at.asc())
                .limit(limit)
                .all()
            )
        return [{"role": row.role, "content": row.content} for row in rows]

    def _append_conversation_turns_sync(
        self, conversation_id: str, question: str, answer: str
    ) -> None:
        with Session(self._engine) as session:
            session.add(ConversationHistory(
                id=str(__import__("uuid").uuid4()),
                conversation_id=conversation_id,
                role="user",
                content=question,
            ))
            session.add(ConversationHistory(
                id=str(__import__("uuid").uuid4()),
                conversation_id=conversation_id,
                role="assistant",
                content=answer,
            ))
            session.commit()
        logger.debug("append_conversation_turns: conversation_id=%s", conversation_id)

    def _get_corpus_stats_sync(self) -> CorpusStats:
        stats = CorpusStats()
        with Session(self._engine) as session:
            stats.total_chunks = session.execute(
                text("SELECT COUNT(*) FROM chunks")
            ).scalar() or 0
            stats.total_papers = session.execute(
                text("SELECT COUNT(*) FROM papers")
            ).scalar() or 0

            for row in session.execute(
                text("SELECT topic, COUNT(*) FROM papers WHERE topic IS NOT NULL GROUP BY topic")
            ):
                stats.by_topic[row[0]] = row[1]

            for row in session.execute(
                text("SELECT year, COUNT(*) FROM papers WHERE year IS NOT NULL GROUP BY year ORDER BY year")
            ):
                stats.by_year[row[0]] = row[1]

            for row in session.execute(
                text("SELECT study_design, COUNT(*) FROM papers WHERE study_design IS NOT NULL GROUP BY study_design")
            ):
                stats.by_study_design[row[0]] = row[1]

        return stats
