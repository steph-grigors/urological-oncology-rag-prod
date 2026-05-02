"""Initial schema: papers, chunks, audit_log, api_keys, conversation_history

Revision ID: 001
Revises:
Create Date: 2026-05-02
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "papers",
        sa.Column("pmc_id", sa.String(20), primary_key=True),
        sa.Column("pmid", sa.String(20)),
        sa.Column("doi", sa.String(100)),
        sa.Column("title", sa.Text),
        sa.Column("abstract", sa.Text),
        sa.Column("journal", sa.String(300)),
        sa.Column("year", sa.Integer),
        sa.Column("topic", sa.String(50)),
        sa.Column("authors", sa.JSON),
        sa.Column("study_design", sa.String(50)),
        sa.Column("cancer_subtype", sa.String(100)),
        sa.Column("patient_population", sa.Text),
        sa.Column("intervention", sa.Text),
        sa.Column("comparator", sa.Text),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now()),
    )

    op.create_table(
        "chunks",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("pmc_id", sa.String(20), sa.ForeignKey("papers.pmc_id"), nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("section_name", sa.String(100)),
        sa.Column("section_type", sa.String(50)),
        sa.Column("chunk_index", sa.Integer),
        sa.Column("total_chunks", sa.Integer),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index("ix_chunks_pmc_id", "chunks", ["pmc_id"])
    # Full-text search column (Postgres only)
    op.execute(
        "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS tsvector_col TSVECTOR "
        "GENERATED ALWAYS AS (to_tsvector('english', coalesce(text, ''))) STORED"
    )
    op.execute("CREATE INDEX IF NOT EXISTS ix_chunks_tsvector ON chunks USING GIN (tsvector_col)")

    op.create_table(
        "audit_log",
        sa.Column("query_id", sa.String(36), primary_key=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("question", sa.Text),
        sa.Column("rewritten_query", sa.Text),
        sa.Column("answer", sa.Text),
        sa.Column("confidence", sa.Float),
        sa.Column("gate_decision", sa.String(20)),
        sa.Column("model", sa.String(100)),
        sa.Column("provider", sa.String(20)),
        sa.Column("input_tokens", sa.Integer),
        sa.Column("output_tokens", sa.Integer),
        sa.Column("latency_ms", sa.Float),
        sa.Column("sources", sa.JSON),
        sa.Column("user_id", sa.String(100)),
        sa.Column("session_id", sa.String(100)),
        sa.Column("hallucinated_citations", sa.JSON),
        sa.Column("flagged", sa.Boolean, server_default="false"),
    )
    op.create_index("ix_audit_log_timestamp", "audit_log", ["timestamp"])
    op.create_index("ix_audit_log_session_id", "audit_log", ["session_id"])

    op.create_table(
        "api_keys",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("key_hash", sa.String(64), unique=True, nullable=False),
        sa.Column("user_id", sa.String(100)),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("is_active", sa.Boolean, server_default="true"),
    )

    op.create_table(
        "conversation_history",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("conversation_id", sa.String(36), nullable=False),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index("ix_conversation_history_id", "conversation_history", ["conversation_id"])


def downgrade() -> None:
    op.drop_table("conversation_history")
    op.drop_table("api_keys")
    op.drop_table("audit_log")
    op.drop_table("chunks")
    op.drop_table("papers")
