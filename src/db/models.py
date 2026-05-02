"""
SQLAlchemy ORM models for all application tables.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Paper(Base):
    __tablename__ = "papers"

    pmc_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    pmid: Mapped[str | None] = mapped_column(String(20))
    doi: Mapped[str | None] = mapped_column(String(100))
    title: Mapped[str | None] = mapped_column(Text)
    abstract: Mapped[str | None] = mapped_column(Text)
    journal: Mapped[str | None] = mapped_column(String(300))
    year: Mapped[int | None] = mapped_column(Integer)
    topic: Mapped[str | None] = mapped_column(String(50))
    authors: Mapped[dict | None] = mapped_column(JSON)
    study_design: Mapped[str | None] = mapped_column(String(50))
    cancer_subtype: Mapped[str | None] = mapped_column(String(100))
    patient_population: Mapped[str | None] = mapped_column(Text)
    intervention: Mapped[str | None] = mapped_column(Text)
    comparator: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    chunks: Mapped[list["Chunk"]] = relationship("Chunk", back_populates="paper")


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    pmc_id: Mapped[str] = mapped_column(ForeignKey("papers.pmc_id"), index=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    section_name: Mapped[str | None] = mapped_column(String(100))
    section_type: Mapped[str | None] = mapped_column(String(50))
    chunk_index: Mapped[int | None] = mapped_column(Integer)
    total_chunks: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    paper: Mapped["Paper"] = relationship("Paper", back_populates="chunks")

    # ix_chunks_pmc_id is created automatically by index=True on the pmc_id column.


class AuditLog(Base):
    __tablename__ = "audit_log"

    query_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    question: Mapped[str | None] = mapped_column(Text)
    rewritten_query: Mapped[str | None] = mapped_column(Text)
    answer: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[float | None] = mapped_column(Float)
    gate_decision: Mapped[str | None] = mapped_column(String(20))
    model: Mapped[str | None] = mapped_column(String(100))
    provider: Mapped[str | None] = mapped_column(String(20))
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    latency_ms: Mapped[float | None] = mapped_column(Float)
    sources: Mapped[list | None] = mapped_column(JSON)
    user_id: Mapped[str | None] = mapped_column(String(100))
    session_id: Mapped[str | None] = mapped_column(String(100))
    hallucinated_citations: Mapped[list | None] = mapped_column(JSON)
    flagged: Mapped[bool] = mapped_column(Boolean, default=False)

    __table_args__ = (
        Index("ix_audit_log_timestamp", "timestamp"),
        Index("ix_audit_log_session_id", "session_id"),
    )


class ApiKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    user_id: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


class ConversationHistory(Base):
    __tablename__ = "conversation_history"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(String(36), index=True)
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
