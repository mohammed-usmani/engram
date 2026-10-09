"""Layer 1: data health. Plain queries over what's stored, no LLM, cheap enough to run nightly.

Each check returns {id, title, status: ok|warn|bad, value, detail, items, href}. The score starts at 100;
a warning costs 5, a problem 15.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

from sqlalchemy import text as sql
from sqlalchemy.ext.asyncio import AsyncSession

from src import dates
from src.memory import facts

PENALTY = {"ok": 0, "warn": 5, "bad": 15}
_NUM = re.compile(r"\d+(?:\.\d+)?")
# past tense about the user's own situation, not "(formerly NetworkChains)" or "previously told X"
_PAST = re.compile(r"\b(previously|formerly|used to|no longer)\s+(work|was|were|employed|live|lived|stud|attend|is|had)", re.I)


def _nums(text: str) -> set[float]:
    return {float(n) for n in _NUM.findall(text or "")}
CONFLICT_SIM = 0.82  # measured on real data: the CGPA 8.0 / 8.27 pair scores ~0.9, unrelated facts with numbers < 0.75


def _check(id: str, title: str, status: str, value, detail: str, items: list | None = None, href: str = "") -> dict:
    return {"id": id, "title": title, "status": status, "value": value, "detail": detail,
            "items": (items or [])[:25], "href": href}


_JUDGE = """Each numbered pair has two statements stored about the same user. Decide if they CONTRADICT:
the SAME attribute of the SAME thing at the SAME time is given two incompatible values, so one of them must be wrong.
Contradictions: two different CGPAs for one degree; two current salaries; "works at X" vs "previously worked at X";
two spellings of the user's own surname.
NOT contradictions: partial or overlapping lists (two descriptions of a tech stack); figures for different companies,
applications, projects or products; an expectation vs a current value; a plan vs a fact; different points in time;
one statement being more detailed than the other.
Return JSON: [{{"pair": <number>, "contradict": true|false, "why": "<at most 12 words>"}}]

{pairs}"""


async def _judge(session: AsyncSession, pairs: list[dict]) -> None:
    """One batched LLM call for pairs not judged before; verdicts are cached in settings (health_verdicts)."""
    from src import settings_store as store
    from src.memory.llm import complete_json
    cache = store.get("health_verdicts") or {}
    todo = [p for p in pairs if f"{p['a_id']}|{p['b_id']}" not in cache]
    for i in range(0, len(todo), 40):
        chunk = todo[i:i + 40]
        listing = "\n".join(f"{n}. A: {p['a']}\n   B: {p['b']}" for n, p in enumerate(chunk))
        try:
            verdicts = await complete_json(_JUDGE.format(pairs=listing), task="reconcile")
        except Exception:
            return  # judged next run; unjudged pairs stay "possible"
        for v in verdicts if isinstance(verdicts, list) else []:
            n = v.get("pair") if isinstance(v, dict) else None
            if isinstance(n, int) and 0 <= n < len(chunk):
                cache[f"{chunk[n]['a_id']}|{chunk[n]['b_id']}"] = {"c": bool(v.get("contradict")), "why": str(v.get("why") or "")[:120]}
    await store.put(session, "health_verdicts", cache)
    for p in pairs:
        v = cache.get(f"{p['a_id']}|{p['b_id']}")
        p["verdict"] = None if v is None else v["c"]
        p["judge"] = v["why"] if v else ""


async def contradictions(session: AsyncSession, judge: bool = True) -> dict:
    """Pairs of near-identical current facts that disagree on a number (8.0 vs 8.27) or on tense
    ("works at" vs "previously worked at")."""
    t = facts.table()
    rows = (await session.execute(sql(f"""
        select id::text, payload->>'data' txt from {t}
        where payload->>'superseded_by' is null and coalesce(payload->>'kind', 'fact') in ('fact', 'preference')
          and length(payload->>'data') < 400"""))).all()
    pairs, seen = [], set()
    for fid, txt in rows:
        near = (await session.execute(sql(f"""
            select b.id::text, b.payload->>'data', 1 - (b.vector <=> a.vector) sim
            from {t} a, {t} b
            where a.id = cast(:id as uuid) and b.id <> a.id and b.payload->>'superseded_by' is null
              and coalesce(b.payload->>'kind', 'fact') in ('fact', 'preference')
            order by b.vector <=> a.vector limit 3"""), {"id": fid})).all()
        for oid, otxt, sim in near:
            if sim < CONFLICT_SIM or (oid, fid) in seen:
                continue
            na, nb = _nums(txt), _nums(otxt)
            numbers = bool(na and nb and na != nb and not (na <= nb or nb <= na))
            tense = bool(_PAST.search(txt)) != bool(_PAST.search(otxt or ""))
            if numbers or tense:
                seen.add((fid, oid))
                pairs.append({"a": txt, "b": otxt, "a_id": fid, "b_id": oid, "similarity": round(float(sim), 2),
                              "why": "different numbers" if numbers else "past vs present"})
    if judge and pairs:
        await _judge(session, pairs)
    real = [p for p in pairs if p.get("verdict") is True]
    unsure = [p for p in pairs if p.get("verdict") is None]
    status = "bad" if len(real) > 3 else "warn" if real or unsure else "ok"
    detail = (f"{len(real)} confirmed" + (f", {len(unsure)} not checked yet" if unsure else "")
              + f" (of {len(pairs)} look-alike pairs with different numbers or tense)") if pairs else "No contradicting facts found"
    return _check("contradictions", "Facts that contradict each other", status, len(real), detail, real + unsure, "/facts")


async def duplicates_and_names(session: AsyncSession) -> list[dict]:
    from src.memory import cleanup
    data = await cleanup._load(session)
    merges = await cleanup.merge_groups(session, data)
    dupes = await cleanup.dupe_groups(session)
    once = await cleanup.once(session, data)
    total = len(data[0]) or 1
    share = len(once) / total
    return [
        _check("duplicate_events", "Duplicate events", "bad" if len(dupes) > 20 else "warn" if dupes else "ok", len(dupes),
               f"{len(dupes)} groups of events that look like the same thing", [
                   {"ids": [e["id"] for e in g.get("items", [])], "text": (g.get("items") or [{}])[0].get("summary", "")}
                   for g in dupes[:25]], "/cleanup#dupes"),
        _check("two_names", "Same thing under two topic names", "bad" if len(merges) > 40 else "warn" if merges else "ok",
               len(merges), f"{len(merges)} groups of topics that look like one", [
                   {"names": [i.get("name") for i in g.get("items", [])]} for g in merges[:25]], "/cleanup#merge"),
        _check("one_off_topics", "Topics mentioned once", "bad" if share > 0.6 else "warn" if share > 0.3 else "ok",
               f"{share:.0%}", f"{len(once)} of {total} topics have one event or none", [], "/cleanup#once"),
    ]


async def date_checks(session: AsyncSession) -> list[dict]:
    _, unreadable = await dates.upcoming(session, 366)
    hidden = await dates.untracked(session, limit=50)
    conflicts = [h for h in hidden if h.get("weekday_conflict")]
    return [
        _check("unreadable_dates", "Date fields Engram can't read", "warn" if unreadable else "ok", len(unreadable),
               f"{len(unreadable)} date fields aren't tracked" if unreadable else "Every date field parses", unreadable,
               "/admin?has=dates"),
        _check("weekday_conflicts", "Dates whose weekday is wrong", "bad" if conflicts else "ok", len(conflicts),
               f"{len(conflicts)} dates say the wrong weekday (e.g. 'Wednesday 8 Oct')" if conflicts else "No weekday mismatches",
               conflicts, "/admin"),
        _check("untracked_dates", "Deadlines buried in document text", "warn" if hidden else "ok", len(hidden),
               f"{len(hidden)} dates inside document text aren't tracked" if hidden else "None", hidden, "/memory"),
    ]


async def fact_size(session: AsyncSession) -> dict:
    rows = (await session.execute(sql(f"""select id::text, left(payload->>'data', 160), length(payload->>'data') n
        from {facts.table()} where payload->>'superseded_by' is null and coalesce(payload->>'kind', 'fact') in ('fact', 'preference')
          and length(payload->>'data') > {facts.PROFILE_FACT_MAX_CHARS} order by n desc"""))).all()
    return _check("oversized_facts", "Oversized facts", "warn" if rows else "ok", len(rows),
                  f"{len(rows)} facts are longer than {facts.PROFILE_FACT_MAX_CHARS} characters (left out of your profile)",
                  [{"id": i, "text": t, "chars": n} for i, t, n in rows], "/facts")


async def pipeline(session: AsyncSession) -> list[dict]:
    jobs = dict((await session.execute(sql("""select status, count(*) from ingest_jobs group by status"""))).all())
    stuck = await session.scalar(sql("""select count(*) from ingest_jobs where status in ('pending', 'processing')
        and updated_at < now() - interval '1 hour'"""))
    r = (await session.execute(sql("""select count(*) n,
            count(*) filter (where flags ? 'empty_brief') empty, count(*) filter (where flags ? 'planner_timeout') timeouts,
            percentile_cont(0.5) within group (order by duration_ms) p50,
            percentile_cont(0.95) within group (order by duration_ms) p95
        from eval_traces where kind = 'recall' and source not in ('eval', 'inspector') and created_at > now() - interval '7 days'"""))).mappings().one()
    x = (await session.execute(sql("""select count(*) n, count(*) filter (where flags ? 'failed') failed,
            count(*) filter (where flags ? 'fallback_used') fallback, count(*) filter (where flags ? 'dropped_events') dropped
        from eval_traces where kind = 'extract' and created_at > now() - interval '7 days'"""))).mappings().one()
    out = [_check("queue", "Extraction queue", "bad" if jobs.get("failed") else "warn" if stuck else "ok",
                  f"{jobs.get('failed', 0)} failed", f"{jobs.get('failed', 0)} failed, {stuck or 0} stuck for over an hour",
                  [], "/queue?status=failed")]
    if r["n"]:
        empty_share = r["empty"] / r["n"]
        out.append(_check("recall_quality", "Recall in production (7 days)",
                          "bad" if empty_share > 0.1 or (r["p95"] or 0) > 15000 else "warn" if empty_share > 0.02 or r["timeouts"] else "ok",
                          f"{r['n']} recalls", f"{r['empty']} came back with no memories, {r['timeouts']} planner timeouts, "
                          f"median {int(r['p50'] or 0)} ms, slowest 5% over {int(r['p95'] or 0)} ms", [], "/traces?kind=recall"))
    if x["n"]:
        out.append(_check("extraction_quality", "Extraction in production (7 days)",
                          "bad" if x["failed"] / x["n"] > 0.1 else "warn" if x["failed"] or x["fallback"] else "ok",
                          f"{x['n']} jobs", f"{x['failed']} failed, {x['fallback']} needed a fallback model, "
                          f"{x['dropped']} had invented events removed", [], "/traces?kind=extract"))
    return out


async def backups() -> dict:
    from src.ui.templating import last_backup
    b = last_backup()
    hours = (datetime.now(timezone.utc) - b).total_seconds() / 3600 if b else None
    return _check("backup", "Backups", "bad" if hours is None or hours > 72 else "warn" if hours > 36 else "ok",
                  f"{int(hours)} h ago" if hours is not None else "never",
                  "Last encrypted backup " + (f"{int(hours)} hours ago" if hours is not None else "not found"), [],
                  "/settings#h-backup")


async def run(session: AsyncSession) -> tuple[float, list[dict]]:
    checks = [await contradictions(session), *await duplicates_and_names(session), *await date_checks(session),
              await fact_size(session), *await pipeline(session), await backups()]
    score = max(0.0, 100.0 - sum(PENALTY[c["status"]] for c in checks))
    return score, checks
