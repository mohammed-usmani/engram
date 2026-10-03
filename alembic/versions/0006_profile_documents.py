"""profile documents: what each public profile (LinkedIn, Indeed, GitHub...) currently says

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-03
"""
from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TYPE doc_type_enum ADD VALUE IF NOT EXISTS 'profile'")


def downgrade() -> None:
    pass  # Postgres can't drop an enum value; an unused 'profile' is harmless
