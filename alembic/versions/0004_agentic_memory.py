"""agentic memory tables (additive only)

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-01
"""
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


_DDL = """
    CREATE TABLE IF NOT EXISTS entities (
        id SERIAL PRIMARY KEY,
        slug VARCHAR(255) NOT NULL UNIQUE,
        kind VARCHAR(50) NOT NULL,
        name VARCHAR(255) NOT NULL,
        aliases JSONB NOT NULL DEFAULT '[]',
        digest TEXT,
        dirty BOOLEAN NOT NULL DEFAULT true,
        embedding vector(768),
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    CREATE TABLE IF NOT EXISTS episodes (
        id SERIAL PRIMARY KEY,
        occurred_at TIMESTAMPTZ NOT NULL,
        kind VARCHAR(50) NOT NULL,
        summary TEXT NOT NULL,
        outcome VARCHAR(100),
        sentiment DOUBLE PRECISION,
        importance INTEGER NOT NULL DEFAULT 3,
        entities JSONB NOT NULL DEFAULT '[]',
        payload JSONB NOT NULL DEFAULT '{}',
        source_agent VARCHAR(100),
        session_id VARCHAR(255),
        embedding vector(768),
        search_vector tsvector GENERATED ALWAYS AS (to_tsvector('english', summary)) STORED,
        access_count INTEGER NOT NULL DEFAULT 0,
        last_accessed TIMESTAMPTZ,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    CREATE INDEX IF NOT EXISTS ix_episodes_occurred_at ON episodes (occurred_at);
    CREATE INDEX IF NOT EXISTS ix_episodes_kind ON episodes (kind);
    CREATE INDEX IF NOT EXISTS ix_episodes_entities_gin ON episodes USING gin (entities);
    CREATE INDEX IF NOT EXISTS ix_episodes_search_vector ON episodes USING gin (search_vector);
    CREATE TABLE IF NOT EXISTS reflections (
        id SERIAL PRIMARY KEY,
        lesson TEXT NOT NULL,
        entities JSONB NOT NULL DEFAULT '[]',
        evidence JSONB NOT NULL DEFAULT '[]',
        confidence DOUBLE PRECISION NOT NULL DEFAULT 0.5,
        embedding vector(768),
        access_count INTEGER NOT NULL DEFAULT 0,
        last_accessed TIMESTAMPTZ,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    CREATE TABLE IF NOT EXISTS session_state (
        id SERIAL PRIMARY KEY,
        session_id VARCHAR(255) NOT NULL,
        key VARCHAR(255) NOT NULL,
        value TEXT NOT NULL,
        expires_at TIMESTAMPTZ NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        CONSTRAINT uq_session_state_session_id UNIQUE (session_id, key)
    );
    CREATE TABLE IF NOT EXISTS memory_blocks (
        name VARCHAR(100) PRIMARY KEY,
        content TEXT NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    CREATE TABLE IF NOT EXISTS ingest_jobs (
        id SERIAL PRIMARY KEY,
        content_hash VARCHAR(64) NOT NULL UNIQUE,
        text TEXT NOT NULL,
        agent VARCHAR(100),
        session_id VARCHAR(255),
        occurred_at TIMESTAMPTZ NOT NULL,
        status VARCHAR(20) NOT NULL DEFAULT 'pending',
        attempts INTEGER NOT NULL DEFAULT 0,
        error TEXT,
        result JSONB NOT NULL DEFAULT '{}',
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    CREATE INDEX IF NOT EXISTS ix_ingest_jobs_status ON ingest_jobs (status);
"""


def upgrade() -> None:
    # asyncpg rejects multi-statement strings; run one at a time
    for stmt in _DDL.split(";"):
        if stmt.strip():
            op.execute(stmt)


def downgrade() -> None:
    # Deliberately a no-op: these tables hold personal memory; drop by hand if ever needed.
    pass
