"""Layer 2: golden questions. Each must find certain text in the recall brief and must not find other text
(e.g. "3.9" but not "3.6"). Fast recall (no LLM), so it costs nothing to run nightly."""
from __future__ import annotations

import re
import time

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.eval.models import GoldenQuestion
from src.memory import recall as rc


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").replace("‑", "-").replace("–", "-")).lower()


def _contains(brief: str, needle: str) -> bool:
    """Case/space-insensitive; a number must not match inside a longer one ("8.0" is not found in "8.07")."""
    b, n = _norm(brief), _norm(needle).strip()
    if not n:
        return True
    pat = re.escape(n)
    if re.fullmatch(r"[\d.]+", n):
        pat = rf"(?<![\d.]){pat}(?![\d])"
    return re.search(pat, b) is not None


async def check(session: AsyncSession, q: GoldenQuestion) -> dict:
    t = time.monotonic()
    out = await rc.recall(session, q.situation, q.budget, fast=True, explain=True, source="eval")
    ms = int((time.monotonic() - t) * 1000)
    brief = out["brief"]
    missing = [m for m in q.must if not _contains(brief, m)]
    forbidden = [m for m in q.must_not if _contains(brief, m)]
    x = out["explain"]
    pinned = sum(s["tokens"] for s in x["sections"] if s["pinned"])
    return {"id": q.id, "situation": q.situation, "passed": not missing and not forbidden, "missing": missing,
            "forbidden": forbidden, "must": q.must, "must_not": q.must_not, "used": x["used"], "pinned": pinned,
            "items": len(out["items"]), "ms": ms, "trace_id": out.get("trace_id")}


async def run(session: AsyncSession) -> tuple[float | None, list[dict], dict]:
    qs = list((await session.execute(select(GoldenQuestion).where(GoldenQuestion.active).order_by(GoldenQuestion.id))).scalars())
    results = [await check(session, q) for q in qs]
    if not results:
        return None, [], {"questions": 0}
    n = len(results)
    must_total = sum(len(r["must"]) for r in results) or 1
    ms = sorted(r["ms"] for r in results)
    summary = {"questions": n, "passed": sum(r["passed"] for r in results),
               "must_hit": round(100 * (1 - sum(len(r["missing"]) for r in results) / must_total), 1),
               "forbidden_hits": sum(len(r["forbidden"]) for r in results),
               "pinned_share": round(100 * sum(r["pinned"] for r in results) / max(sum(r["used"] for r in results), 1), 1),
               "median_ms": ms[n // 2], "max_ms": ms[-1]}
    return round(100 * summary["passed"] / n, 1), results, summary
