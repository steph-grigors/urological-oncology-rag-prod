"""
Application settings loaded from environment variables via Pydantic BaseSettings.

All runtime configuration lives here. Instantiate `get_settings()` once and
pass the returned object through dependency injection — never read os.environ
directly in application code.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── API keys ────────────────────────────────────────────────────────────
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field("", alias="ANTHROPIC_API_KEY")
    cohere_api_key: str = Field("", alias="COHERE_API_KEY")

    # ── Observability ────────────────────────────────────────────────────────
    langfuse_public_key: str = Field("", alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = Field("", alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field("https://cloud.langfuse.com", alias="LANGFUSE_HOST")

    # ── Vector store (Qdrant) ────────────────────────────────────────────────
    qdrant_url: str = Field("http://localhost:6333", alias="QDRANT_URL")
    qdrant_api_key: str = Field("", alias="QDRANT_API_KEY")
    qdrant_collection: str = Field(
        "urological_oncology_papers", alias="QDRANT_COLLECTION"
    )

    # ── Document store (Postgres) ────────────────────────────────────────────
    postgres_url: str = Field(
        "postgresql+asyncpg://rag:rag@localhost:5432/rag", alias="POSTGRES_URL"
    )

    # ── Model selection ──────────────────────────────────────────────────────
    embedding_model: str = Field(
        "text-embedding-3-small", alias="EMBEDDING_MODEL"
    )
    generation_model: str = Field("claude-sonnet-4-6", alias="GENERATION_MODEL")
    generation_provider: Literal["anthropic", "openai"] = Field(
        "anthropic", alias="GENERATION_PROVIDER"
    )
    metadata_extraction_model: str = Field(
        "gpt-4o-mini", alias="METADATA_EXTRACTION_MODEL"
    )

    # ── Retrieval ────────────────────────────────────────────────────────────
    top_k_retrieval: int = Field(20, alias="TOP_K_RETRIEVAL", ge=1, le=100)
    top_k_rerank: int = Field(5, alias="TOP_K_RERANK", ge=1, le=50)
    confidence_threshold: float = Field(
        0.45, alias="CONFIDENCE_THRESHOLD", ge=0.0, le=1.0
    )
    max_context_chars: int = Field(6000, alias="MAX_CONTEXT_CHARS", ge=500)

    # ── Chunking ─────────────────────────────────────────────────────────────
    chunk_size_words: int = Field(200, alias="CHUNK_SIZE_WORDS", ge=50, le=1000)
    chunk_overlap_words: int = Field(30, alias="CHUNK_OVERLAP_WORDS", ge=0, le=200)

    # ── Application ──────────────────────────────────────────────────────────
    app_env: Literal["development", "staging", "production"] = Field(
        "development", alias="APP_ENV"
    )
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    api_key_header: str = Field("X-API-Key", alias="API_KEY_HEADER")
    api_keys: list[str] = Field(default_factory=list, alias="API_KEYS")
    admin_api_key: str = Field("", alias="ADMIN_API_KEY")
    rate_limit_per_minute: int = Field(60, alias="RATE_LIMIT_PER_MINUTE", ge=1)
    cors_origins: list[str] = Field(default_factory=list, alias="CORS_ORIGINS")

    @field_validator("chunk_overlap_words")
    @classmethod
    def overlap_less_than_chunk(cls, v: int, info) -> int:
        chunk_size = info.data.get("chunk_size_words", 200)
        if v >= chunk_size:
            raise ValueError(
                f"chunk_overlap_words ({v}) must be less than chunk_size_words ({chunk_size})"
            )
        return v

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "populate_by_name": True,
        "extra": "ignore",
    }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance. Safe to call anywhere."""
    return Settings()
