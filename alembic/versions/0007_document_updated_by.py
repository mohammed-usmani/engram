"""documents record who last changed them (claude-code, chatgpt, admin...)

Revision ID: 0007
Revises: 0006
Create Date: 2026-10-03
"""
import sqlalchemy as sa
from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("context_documents", sa.Column("updated_by", sa.String(64), nullable=True))


def downgrade() -> None:
    op.drop_column("context_documents", "updated_by")
