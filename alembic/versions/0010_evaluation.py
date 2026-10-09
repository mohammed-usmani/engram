"""evaluation: traces, runs, golden questions — additive

Revision ID: 0010
Revises: 0009
"""
from alembic import op

revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""CREATE TABLE IF NOT EXISTS eval_traces (
        id serial PRIMARY KEY, kind varchar(16) NOT NULL, source varchar(32), input text NOT NULL DEFAULT '',
        data jsonb NOT NULL DEFAULT '{}', flags jsonb NOT NULL DEFAULT '[]', duration_ms integer NOT NULL DEFAULT 0,
        feedback smallint, note text, created_at timestamptz NOT NULL DEFAULT now())""")
    op.execute("CREATE INDEX IF NOT EXISTS ix_eval_traces_kind ON eval_traces (kind)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_eval_traces_source ON eval_traces (source)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_eval_traces_created_at ON eval_traces (created_at)")
    op.execute("""CREATE TABLE IF NOT EXISTS eval_runs (
        id serial PRIMARY KEY, kind varchar(16) NOT NULL, trigger varchar(32) NOT NULL DEFAULT 'manual',
        score double precision, summary jsonb NOT NULL DEFAULT '{}', details jsonb NOT NULL DEFAULT '[]',
        started_at timestamptz NOT NULL DEFAULT now(), finished_at timestamptz)""")
    op.execute("CREATE INDEX IF NOT EXISTS ix_eval_runs_kind ON eval_runs (kind)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_eval_runs_started_at ON eval_runs (started_at)")
    op.execute("""CREATE TABLE IF NOT EXISTS eval_questions (
        id serial PRIMARY KEY, situation text NOT NULL, must jsonb NOT NULL DEFAULT '[]',
        must_not jsonb NOT NULL DEFAULT '[]', budget integer NOT NULL DEFAULT 1500, note text, trace_id integer,
        active boolean NOT NULL DEFAULT true, created_at timestamptz NOT NULL DEFAULT now())""")


def downgrade() -> None:
    pass  # additive; dropping would delete evaluation history
