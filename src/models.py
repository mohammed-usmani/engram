from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Enum, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column
from pgvector.sqlalchemy import Vector

from .database import Base


class DocType(str, enum.Enum):
    RESUME = "resume"
    PROJECT = "project"
    SKILL = "skill"
    EXPERIENCE = "experience"
    EDUCATION = "education"
    ACHIEVEMENT = "achievement"
    CERTIFICATION = "certification"
    PROFILE = "profile"  # what a public profile (LinkedIn, Indeed, GitHub...) currently says


class Source(str, enum.Enum):
    FILE = "file"
    MANUAL = "manual"


class ContextDocument(Base):
    __tablename__ = "context_documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    type: Mapped[DocType] = mapped_column(
        Enum(DocType, name="doc_type_enum", values_callable=lambda x: [e.value for e in x]),
        index=True,
        nullable=False,
    )
    slug: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    tags: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    sections: Mapped[dict[str, str]] = mapped_column(JSONB, nullable=False, default=dict)
    source: Mapped[Source] = mapped_column(
        Enum(Source, name="source_enum", values_callable=lambda x: [e.value for e in x]),
        index=True,
        nullable=False,
        default=Source.MANUAL,
    )
    doc_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )
    search_vector: Mapped[Any] = mapped_column(TSVECTOR, nullable=True)
    embedding: Mapped[Any] = mapped_column(Vector(768), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_context_documents_tags_gin", "tags", postgresql_using="gin"),
        Index("ix_context_documents_search_vector", "search_vector", postgresql_using="gin"),
    )


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str | None] = mapped_column(String(512))
    summary: Mapped[str | None] = mapped_column(Text)
    message_count: Mapped[int] = mapped_column(Integer, default=0)
    provider: Mapped[str | None] = mapped_column(String(50), default="ollama")
    model: Mapped[str | None] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class MessageRole(str, enum.Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True, nullable=False
    )
    role: Mapped[MessageRole] = mapped_column(
        Enum(MessageRole, name="message_role_enum", values_callable=lambda x: [e.value for e in x]),
        nullable=False,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class MemoryCategory(str, enum.Enum):
    FACT = "fact"
    PREFERENCE = "preference"
    EXPERIENCE = "experience"
    SKILL = "skill"


class Memory(Base):
    __tablename__ = "memories"

    id: Mapped[int] = mapped_column(primary_key=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[MemoryCategory] = mapped_column(
        Enum(MemoryCategory, name="memory_category_enum", values_callable=lambda x: [e.value for e in x]),
        nullable=False,
    )
    embedding: Mapped[Any] = mapped_column(Vector(768), nullable=True)
    source_session_id: Mapped[int | None] = mapped_column(
        ForeignKey("chat_sessions.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_memories_category", "category"),
    )


class ProviderSetting(Base):
    __tablename__ = "provider_settings"

    id: Mapped[int] = mapped_column(primary_key=True)
    provider: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    api_key: Mapped[str | None] = mapped_column(Text)
    model: Mapped[str | None] = mapped_column(String(200))
    base_url: Mapped[str | None] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
