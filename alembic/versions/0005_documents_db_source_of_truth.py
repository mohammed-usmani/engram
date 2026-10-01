"""documents are edited on the site: the database is the source of truth

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-01
"""
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Files are now only an import path; existing documents must never be overwritten by one.
    op.execute("UPDATE context_documents SET source = 'manual' WHERE source = 'file'")


def downgrade() -> None:
    pass  # nothing to undo safely: we can't know which rows were file-backed
