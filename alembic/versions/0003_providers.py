"""add provider settings and session provider columns

Revision ID: 0003
Revises: 0002
Create Date: 2026-04-16
"""
from alembic import op
import sqlalchemy as sa

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None

_DEFAULT_PROVIDERS = [
    ("ollama", None, None, None, True),
    ("groq", None, "llama-3.3-70b-versatile", "https://api.groq.com/openai/v1", False),
    ("cerebras", None, "llama3.1-70b", "https://api.cerebras.ai/v1", False),
    ("openai", None, "gpt-4o-mini", "https://api.openai.com/v1", False),
    ("gemini", None, "gemini-2.0-flash", None, False),
    ("claude", None, "claude-sonnet-4-20250514", None, False),
]


def upgrade() -> None:
    op.create_table(
        "provider_settings",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("provider", sa.String(50), unique=True, nullable=False),
        sa.Column("api_key", sa.Text),
        sa.Column("model", sa.String(200)),
        sa.Column("base_url", sa.Text),
        sa.Column("is_active", sa.Boolean, server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    # Seed default rows
    provider_settings = sa.table(
        "provider_settings",
        sa.column("provider", sa.String),
        sa.column("api_key", sa.Text),
        sa.column("model", sa.String),
        sa.column("base_url", sa.Text),
        sa.column("is_active", sa.Boolean),
    )
    op.bulk_insert(provider_settings, [
        {"provider": p, "api_key": k, "model": m, "base_url": u, "is_active": a}
        for p, k, m, u, a in _DEFAULT_PROVIDERS
    ])

    op.add_column("chat_sessions", sa.Column("provider", sa.String(50), server_default="ollama"))
    op.add_column("chat_sessions", sa.Column("model", sa.String(200)))


def downgrade() -> None:
    op.drop_column("chat_sessions", "model")
    op.drop_column("chat_sessions", "provider")
    op.drop_table("provider_settings")
