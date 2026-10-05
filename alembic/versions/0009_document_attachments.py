"""document attachments (bytes in the DB, extracted text searchable) — additive

Revision ID: 0009
Revises: 0008
"""
from alembic import op

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE context_documents ADD COLUMN IF NOT EXISTS attachments_text text NOT NULL DEFAULT ''")
    op.execute("""CREATE TABLE IF NOT EXISTS document_attachments (
        id serial PRIMARY KEY,
        document_id integer NOT NULL REFERENCES context_documents(id) ON DELETE CASCADE,
        filename varchar(255) NOT NULL,
        mime varchar(127) NOT NULL,
        size integer NOT NULL,
        sha256 varchar(64) NOT NULL,
        text text NOT NULL DEFAULT '',
        data bytea NOT NULL,
        created_by varchar(64),
        created_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT uq_document_attachments_doc_sha UNIQUE (document_id, sha256))""")
    op.execute("CREATE INDEX IF NOT EXISTS ix_document_attachments_document_id ON document_attachments (document_id)")
    op.execute("""
        CREATE OR REPLACE FUNCTION context_documents_tsv_update() RETURNS trigger AS $$
        BEGIN
          NEW.search_vector :=
            setweight(to_tsvector('english', coalesce(NEW.title, '')), 'A') ||
            setweight(to_tsvector('english', coalesce(NEW.content, '')), 'B') ||
            setweight(to_tsvector('english', coalesce(NEW.attachments_text, '')), 'C');
          RETURN NEW;
        END
        $$ LANGUAGE plpgsql;
    """)


def downgrade() -> None:
    pass  # additive; dropping would delete the user's files
