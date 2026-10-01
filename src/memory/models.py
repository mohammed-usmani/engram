"""Tables for the agentic memory layers. Facts/preferences/procedures live in mem0."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import Computed, DateTime, Float, Index, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column

from src.database import Base


def _ts(**kw):
    return mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False, **kw)


class Entity(Base):
    """Join key across layers: company:acme, topic:job-search, project:cityfix, person:x."""
    __tablename__ = "entities"

    id: Mapped[int] = mapped_column(primary_key=True)
    slug: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    kind: Mapped[str] = mapped_column(String(50), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    aliases: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    digest: Mapped[str | None] = mapped_column(Text)
    dirty: Mapped[bool] = mapped_column(default=True, nullable=False)
    embedding: Mapped[Any] = mapped_column(Vector(768), nullable=True)
    created_at: Mapped[datetime] = _ts()
    updated_at: Mapped[datetime] = _ts(onupdate=func.now())


class Episode(Base):
    """Something that happened, at a time. Aggregates (counts, first/last) run over this."""
    __tablename__ = "episodes"

    id: Mapped[int] = mapped_column(primary_key=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    kind: Mapped[str] = mapped_column(String(50), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    outcome: Mapped[str | None] = mapped_column(String(100))
    sentiment: Mapped[float | None] = mapped_column(Float)
    importance: Mapped[int] = mapped_column(Integer, default=3, nullable=False)
    entities: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    source_agent: Mapped[str | None] = mapped_column(String(100))
    session_id: Mapped[str | None] = mapped_column(String(255))
    embedding: Mapped[Any] = mapped_column(Vector(768), nullable=True)
    search_vector: Mapped[Any] = mapped_column(
        TSVECTOR, Computed("to_tsvector('english', summary)", persisted=True)
    )
    access_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_accessed: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = _ts()

    __table_args__ = (
        Index("ix_episodes_occurred_at", "occurred_at"),
        Index("ix_episodes_kind", "kind"),
        Index("ix_episodes_entities_gin", "entities", postgresql_using="gin"),
        Index("ix_episodes_search_vector", "search_vector", postgresql_using="gin"),
    )


class Reflection(Base):
    """A lesson derived from several episodes, with the evidence that supports it."""
    __tablename__ = "reflections"

    id: Mapped[int] = mapped_column(primary_key=True)
    lesson: Mapped[str] = mapped_column(Text, nullable=False)
    entities: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    evidence: Mapped[list[int]] = mapped_column(JSONB, nullable=False, default=list)
    confidence: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)
    embedding: Mapped[Any] = mapped_column(Vector(768), nullable=True)
    access_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_accessed: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = _ts()
    updated_at: Mapped[datetime] = _ts(onupdate=func.now())


class SessionState(Base):
    """Short-term memory for one ongoing task/session; expires."""
    __tablename__ = "session_state"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[str] = mapped_column(String(255), nullable=False)
    key: Mapped[str] = mapped_column(String(255), nullable=False)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = _ts()

    __table_args__ = (UniqueConstraint("session_id", "key"),)


class MemoryBlock(Base):
    """Small always-included blocks, e.g. the user profile."""
    __tablename__ = "memory_blocks"

    name: Mapped[str] = mapped_column(String(100), primary_key=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = _ts(onupdate=func.now())


class IngestJob(Base):
    """Durable write queue; content_hash makes retries from hooks idempotent."""
    __tablename__ = "ingest_jobs"

    id: Mapped[int] = mapped_column(primary_key=True)
    content_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    agent: Mapped[str | None] = mapped_column(String(100))
    session_id: Mapped[str | None] = mapped_column(String(255))
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="pending", nullable=False, index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error: Mapped[str | None] = mapped_column(Text)
    result: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = _ts()
    updated_at: Mapped[datetime] = _ts(onupdate=func.now())
