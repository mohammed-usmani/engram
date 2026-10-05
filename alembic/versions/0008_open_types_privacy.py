"""open document types (registry) and per-document privacy — additive, no data rewritten

Revision ID: 0008
Revises: 0007
"""
from alembic import op

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None

from src.models import BUILTIN_TYPES  # noqa: E402


def upgrade() -> None:
    # enum -> text: every existing value is kept as the same string
    op.execute("ALTER TABLE context_documents ALTER COLUMN type TYPE varchar(40) USING type::text")
    op.execute("ALTER TABLE context_documents ADD COLUMN IF NOT EXISTS privacy varchar(16) NOT NULL DEFAULT 'normal'")
    op.execute("""DO $$ BEGIN
        ALTER TABLE context_documents ADD CONSTRAINT ck_context_documents_privacy
            CHECK (privacy IN ('normal', 'private', 'local-only'));
        EXCEPTION WHEN duplicate_object THEN NULL; END $$""")
    op.execute("""CREATE TABLE IF NOT EXISTS document_types (
        name varchar(40) PRIMARY KEY,
        description text NOT NULL,
        builtin boolean NOT NULL DEFAULT false,
        created_at timestamptz NOT NULL DEFAULT now())""")
    for name, desc in BUILTIN_TYPES.items():
        op.execute(f"INSERT INTO document_types (name, description, builtin) VALUES "
                   f"('{name}', '{desc.replace(chr(39), chr(39) * 2)}', true) ON CONFLICT (name) DO NOTHING")


def downgrade() -> None:
    pass  # additive; nothing to undo safely
