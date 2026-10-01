"""add chat sessions, messages, memories, embeddings

Revision ID: 0002
Revises: 0001
Create Date: 2026-04-16
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # pgvector extension must be created by a superuser beforehand
    # op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.add_column("context_documents", sa.Column("embedding", Vector(768), nullable=True))
    op.create_index("ix_context_documents_embedding", "context_documents", ["embedding"],
                    postgresql_using="hnsw", postgresql_ops={"embedding": "vector_cosine_ops"})

    # Create enums explicitly, then use create_type=False in columns
    op.execute("CREATE TYPE message_role_enum AS ENUM ('user', 'assistant', 'system')")
    op.execute("CREATE TYPE memory_category_enum AS ENUM ('fact', 'preference', 'experience', 'skill')")

    msg_role = postgresql.ENUM("user", "assistant", "system", name="message_role_enum", create_type=False)
    mem_cat = postgresql.ENUM("fact", "preference", "experience", "skill", name="memory_category_enum", create_type=False)

    op.create_table(
        "chat_sessions",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("title", sa.String(512)),
        sa.Column("summary", sa.Text),
        sa.Column("message_count", sa.Integer, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    op.create_table(
        "chat_messages",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("session_id", sa.Integer, sa.ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("role", msg_role, nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_chat_messages_session_id", "chat_messages", ["session_id"])

    op.create_table(
        "memories",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("category", mem_cat, nullable=False),
        sa.Column("embedding", Vector(768), nullable=True),
        sa.Column("source_session_id", sa.Integer,
                  sa.ForeignKey("chat_sessions.id", ondelete="SET NULL"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_memories_category", "memories", ["category"])
    op.create_index("ix_memories_embedding", "memories", ["embedding"],
                    postgresql_using="hnsw", postgresql_ops={"embedding": "vector_cosine_ops"})


def downgrade() -> None:
    op.drop_table("memories")
    op.drop_table("chat_messages")
    op.drop_table("chat_sessions")
    op.drop_index("ix_context_documents_embedding", table_name="context_documents")
    op.drop_column("context_documents", "embedding")
    sa.Enum(name="memory_category_enum").drop(op.get_bind(), checkfirst=True)
    sa.Enum(name="message_role_enum").drop(op.get_bind(), checkfirst=True)
    op.execute("DROP EXTENSION IF EXISTS vector")
