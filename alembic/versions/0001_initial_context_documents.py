"""initial context_documents

Revision ID: 0001
Revises:
Create Date: 2026-04-16
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create enum types via raw SQL first
    op.execute(
        "CREATE TYPE doc_type_enum AS ENUM "
        "('resume', 'project', 'skill', 'experience', 'education', 'achievement', 'certification')"
    )
    op.execute("CREATE TYPE source_enum AS ENUM ('file', 'manual')")

    # Use postgresql.ENUM with create_type=False so SA doesn't try to re-create them
    doc_type = postgresql.ENUM(
        "resume", "project", "skill", "experience", "education", "achievement", "certification",
        name="doc_type_enum",
        create_type=False,
    )
    source = postgresql.ENUM("file", "manual", name="source_enum", create_type=False)

    op.create_table(
        "context_documents",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("type", doc_type, nullable=False),
        sa.Column("slug", sa.String(255), nullable=False),
        sa.Column("title", sa.String(512), nullable=False),
        sa.Column("tags", postgresql.JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("sections", postgresql.JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("source", source, nullable=False, server_default="manual"),
        sa.Column("metadata", postgresql.JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("search_vector", postgresql.TSVECTOR, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_context_documents_slug", "context_documents", ["slug"], unique=True)
    op.create_index("ix_context_documents_type", "context_documents", ["type"])
    op.create_index("ix_context_documents_source", "context_documents", ["source"])
    op.create_index("ix_context_documents_tags_gin", "context_documents", ["tags"], postgresql_using="gin")
    op.create_index("ix_context_documents_search_vector", "context_documents", ["search_vector"], postgresql_using="gin")

    op.execute("""
        CREATE OR REPLACE FUNCTION context_documents_tsv_update() RETURNS trigger AS $$
        BEGIN
          NEW.search_vector :=
            setweight(to_tsvector('english', coalesce(NEW.title, '')), 'A') ||
            setweight(to_tsvector('english', coalesce(NEW.content, '')), 'B');
          RETURN NEW;
        END
        $$ LANGUAGE plpgsql;
    """)
    op.execute("""
        CREATE TRIGGER context_documents_tsv_trg
        BEFORE INSERT OR UPDATE ON context_documents
        FOR EACH ROW EXECUTE FUNCTION context_documents_tsv_update();
    """)


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS context_documents_tsv_trg ON context_documents")
    op.execute("DROP FUNCTION IF EXISTS context_documents_tsv_update()")
    op.drop_index("ix_context_documents_search_vector", table_name="context_documents")
    op.drop_index("ix_context_documents_tags_gin", table_name="context_documents")
    op.drop_index("ix_context_documents_source", table_name="context_documents")
    op.drop_index("ix_context_documents_type", table_name="context_documents")
    op.drop_index("ix_context_documents_slug", table_name="context_documents")
    op.drop_table("context_documents")
    sa.Enum(name="source_enum").drop(op.get_bind(), checkfirst=True)
    sa.Enum(name="doc_type_enum").drop(op.get_bind(), checkfirst=True)
