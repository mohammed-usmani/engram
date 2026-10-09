"""Tables for evaluation. Traces are kept 30 days; runs and questions are kept."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, Integer, SmallInteger, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from src.database import Base


class Trace(Base):
    """One recall or extraction as it happened in production: input, what came out, timings, red flags."""
    __tablename__ = "eval_traces"

    id: Mapped[int] = mapped_column(primary_key=True)
    kind: Mapped[str] = mapped_column(String(16), index=True)             # recall | extract
    source: Mapped[str | None] = mapped_column(String(32), index=True)    # mcp, rest, ask, inspector, eval, worker...
    input: Mapped[str] = mapped_column(Text, default="")
    data: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    flags: Mapped[list[str]] = mapped_column(JSONB, default=list)          # e.g. empty_brief, planner_timeout
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    feedback: Mapped[int | None] = mapped_column(SmallInteger)            # +1 / -1 from the user
    note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)


class EvalRun(Base):
    """One run of an evaluation layer, with its score and every check's result."""
    __tablename__ = "eval_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    kind: Mapped[str] = mapped_column(String(16), index=True)             # health | recall | extract
    trigger: Mapped[str] = mapped_column(String(32), default="manual")    # manual | nightly | cli
    score: Mapped[float | None] = mapped_column(Float)                    # 0-100
    summary: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    details: Mapped[list[Any]] = mapped_column(JSONB, default=list)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class GoldenQuestion(Base):
    """A question the recall brief must answer: text that must appear, and text that must not."""
    __tablename__ = "eval_questions"

    id: Mapped[int] = mapped_column(primary_key=True)
    situation: Mapped[str] = mapped_column(Text)
    must: Mapped[list[str]] = mapped_column(JSONB, default=list)
    must_not: Mapped[list[str]] = mapped_column(JSONB, default=list)
    budget: Mapped[int] = mapped_column(Integer, default=1500)
    note: Mapped[str | None] = mapped_column(Text)
    trace_id: Mapped[int | None] = mapped_column(Integer)
    active: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
