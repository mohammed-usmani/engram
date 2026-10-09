"""Layer 3: the extraction test set. Fixed, labelled texts scored on what recall depends on: how many events,
what kind, which day, which entities, and that facts / procedures / notes land in the right layer.
Also holds regression cases from real failures (recaps, hypotheticals, plans that aren't applications).

Runs against any provider/model without touching the live configuration (llm.FORCE)."""
from __future__ import annotations

import asyncio
import time
from datetime import datetime

from src.memory import USER_TZ, llm, local_date
from src.memory.extract import CAREER_TOPIC, extract

NOW = datetime(2026, 10, 1, 10, 0, tzinfo=USER_TZ)  # a Thursday


def names(ex) -> set[str]:
    out = {e["name"].lower() for e in ex.entities}
    for item in [*ex.episodes, *ex.facts, *ex.procedures]:
        out |= {n.lower() for n in item["entities"]}
    return out


def has(ex, word: str) -> bool:
    return any(word in n for n in names(ex))


def kinds(ex) -> list[str]:
    return sorted(e["kind"] for e in ex.episodes)


def days(ex) -> list[str]:
    return sorted(local_date(e["occurred_at"]) for e in ex.episodes)


# (name, text, [(check name, predicate)])
CASES = [
    ("applied today", "I applied to Stripe for a backend engineer role today through LinkedIn.", [
        ("1 applied", lambda x: kinds(x) == ["applied"]),
        ("date today", lambda x: days(x) == ["2026-10-01"]),
        ("entity stripe", lambda x: has(x, "stripe")),
    ]),
    ("two companies, relative date", "Applied to Zomato and Swiggy for SDE-2 roles last Monday.", [
        ("2 applied (one per company)", lambda x: kinds(x) == ["applied", "applied"]),
        ("date last Monday", lambda x: days(x) and set(days(x)) == {"2026-09-28"}),
        ("entities both", lambda x: has(x, "zomato") and has(x, "swiggy")),
    ]),
    ("interview yesterday", "Yesterday I had the Razorpay technical round. DSA went great but I struggled with the system design question.", [
        ("1 interview", lambda x: kinds(x) == ["interview"]),
        ("date yesterday", lambda x: days(x) == ["2026-09-30"]),
        ("entity razorpay", lambda x: has(x, "razorpay")),
    ]),
    ("rejection", "Got rejected by Stripe after the recruiter screen, they wanted more distributed systems experience.", [
        ("1 rejection", lambda x: kinds(x) == ["rejection"]),
        ("outcome rejected", lambda x: [e["outcome"] for e in x.episodes] == ["rejected"]),
        ("entity stripe", lambda x: has(x, "stripe")),
    ]),
    ("preferences only", "I prefer short, direct answers, and I use Arch Linux with Hyprland on my laptop.", [
        ("no events", lambda x: kinds(x) == []),
        ("≥1 preference fact", lambda x: any(f["kind"] == "preference" for f in x.facts)),
        ("not tagged job search", lambda x: not any(CAREER_TOPIC.lower() in [n.lower() for n in f["entities"]]
                                                     for f in x.facts)),
    ]),
    ("move last month", "Moved to Bangalore last month for my new job at Example Labs.", [
        ("1 event", lambda x: len(x.episodes) == 1),
        ("date in September", lambda x: len(days(x)) == 1 and days(x)[0].startswith("2026-09")),
        ("fact about Bangalore", lambda x: any("bangalore" in f["text"].lower() for f in x.facts)),
    ]),
    ("non-career event", "Ran a 5k this morning in 28 minutes, feeling great.", [
        ("1 event", lambda x: len(x.episodes) == 1),
        ("date today", lambda x: days(x) == ["2026-10-01"]),
        ("not tagged job search", lambda x: not has(x, "job search")),
    ]),
    ("procedure", "When I deploy CityFix I run the tests, build the docker image, push it to Railway, then check the health endpoint.", [
        ("procedure captured", lambda x: len(x.procedures) >= 1),
        ("≥4 steps", lambda x: bool(x.procedures) and len(x.procedures[0]["steps"]) >= 4),
        ("entity cityfix", lambda x: has(x, "cityfix")),
    ]),
    ("session notes", "For this report: it needs to be a PDF and it's due Friday.", [
        ("no events", lambda x: kinds(x) == []),
        ("session notes", lambda x: len(x.session_notes) >= 1),
        ("mentions PDF", lambda x: any("pdf" in (n["key"] + n["value"]).lower() for n in x.session_notes)),
    ]),
    ("two dated events", "Had the final round at Acme on September 25th and got the offer today!", [
        ("interview + offer", lambda x: kinds(x) == ["interview", "offer"]),
        ("dates 25th and today", lambda x: days(x) == ["2026-09-25", "2026-10-01"]),
        ("entity acme", lambda x: has(x, "acme")),
    ]),
    ("small talk", "What's the weather like today?", [
        ("nothing extracted", lambda x: not x.episodes and not x.facts and not x.procedures),
    ]),
    ("coding session with life events", """[Claude Code session in project jobsearch]
user: can you fix the typst build, the cv won't compile after I changed the font
assistant: The font name needs to be "JetBrainsMono NF". I updated cv/main.typ and the build passes now.
user: nice. btw I sent applications to Cred and Groww this morning, both for backend roles. used the tailored cv
assistant: Want me to log them in applications.md?
user: yes please. also Meesho got back to me, recruiter screen is on Monday
assistant: Logged Cred and Groww as applied today and added Meesho with a screen scheduled for next Monday.
user: thanks. keep commit messages short from now on, I hate long ones""", [
        ("2 applied (Cred, Groww)", lambda x: kinds(x).count("applied") == 2),
        ("applied dated today", lambda x: all(local_date(e["occurred_at"]) == "2026-10-01"
                                              for e in x.episodes if e["kind"] == "applied")),
        ("entities cred, groww, meesho", lambda x: has(x, "cred") and has(x, "groww") and has(x, "meesho")),
        ("coding chatter not stored as events", lambda x: len(x.episodes) <= 4),
        ("commit-message preference", lambda x: any("commit" in f["text"].lower() for f in x.facts)),
    ]),
    ("incident in a session", """[Claude Code session in project cityfix]
user: the deploy to railway failed again with a 502
assistant: The health check hits /health before migrations finish. I added a 20s start period and redeployed; it's healthy now.
user: great, that's the second time this month the migrations caused it
assistant: I can make migrations run in a release step so this can't happen again.
user: do it. also remember I'm on the free railway plan so keep it to one service""", [
        ("incident captured", lambda x: any(e["kind"] in ("incident", "task_done", "milestone") for e in x.episodes)),
        ("entity cityfix", lambda x: has(x, "cityfix")),
        ("railway free-plan fact", lambda x: any("free" in f["text"].lower() and "railway" in f["text"].lower()
                                                 for f in x.facts)),
        ("not tagged job search", lambda x: not has(x, "job search")),
    ]),
    ("person and plan", "My manager Priya asked me to lead the payments migration project starting next week.", [
        ("person priya", lambda x: has(x, "priya")),
        ("something stored", lambda x: bool(x.episodes or x.facts)),
    ]),
    # ---------------------------------------------------------------- regressions from real failures
    ("REG recap: no invented dates", """Consolidated job-application history:

Globex — Backend Engineer, Pune. Strong fit. Tailored resume created.

Initech — AI Engineer, Mumbai. Screening on Sep 28 went well; recruiter asked for notice period.

Umbrella — Platform Engineer. Low priority, PHP stack.""", [
        ("no event dated on a day the text never gives",
         lambda x: all(local_date(e["occurred_at"]) in ("2026-10-01", "2026-09-28") for e in x.episodes)),
        ("Initech screening kept with its date",
         lambda x: any(e["kind"] == "interview" and local_date(e["occurred_at"]) == "2026-09-28" for e in x.episodes)),
    ]),
    ("REG plan is not an application", "Researched Hooli and tailored my resume for their backend role; I'll apply next week.", [
        ("no applied event", lambda x: "applied" not in kinds(x)),
        ("entity hooli", lambda x: has(x, "hooli")),
    ]),
    ("REG quoted example is not an event", """[Claude Code session in project memoryapp]
user: write the README section that explains what the app stores
assistant: Here's the section. For example, if someone tells their assistant "Had my Acme onsite yesterday, system design went badly", the app stores an interview event for Acme and a lesson about system design.
user: looks good, ship it""", [
        ("no Acme interview invented", lambda x: not any(e["kind"] == "interview" for e in x.episodes)),
    ]),
    ("REG reply is not an application", "Northwind replied positively to my email about the backend role and asked for a call on Monday.", [
        ("no applied event", lambda x: "applied" not in kinds(x)),
        ("entity northwind", lambda x: has(x, "northwind")),
    ]),
]


async def _case(name: str, text: str, checks, sem: asyncio.Semaphore) -> dict:
    async with sem:
        t = time.monotonic()
        try:
            ex = await extract(text, NOW)
        except Exception as e:  # unusable output from every provider
            return {"case": name, "passed": 0, "total": len(checks), "failed": [n for n, _ in checks],
                    "error": str(e)[:200], "seconds": None, "output": None}
        secs = round(time.monotonic() - t, 1)
    failed = []
    for check, pred in checks:
        try:
            ok = bool(pred(ex))
        except Exception:
            ok = False
        if not ok:
            failed.append(check)
    return {"case": name, "passed": len(checks) - len(failed), "total": len(checks), "failed": failed,
            "error": None, "seconds": secs,
            "output": {"episodes": [f"{e['kind']} {local_date(e['occurred_at'])} {e['summary']}" for e in ex.episodes],
                       "facts": [f["text"] for f in ex.facts][:10], "dropped": ex.dropped}}


async def run(provider: str, model: str | None, concurrency: int = 4) -> tuple[float, list[dict], dict]:
    """Score one provider/model on every case. Doesn't change what live extraction uses."""
    token = llm.FORCE.set([(provider, model)])
    try:
        sem = asyncio.Semaphore(concurrency)
        results = list(await asyncio.gather(*(_case(n, t, c, sem) for n, t, c in CASES)))
    finally:
        llm.FORCE.reset(token)
    passed = sum(r["passed"] for r in results)
    total = sum(r["total"] for r in results)
    secs = sorted(r["seconds"] for r in results if r["seconds"] is not None)
    summary = {"provider": provider, "model": model or llm.default_model(provider), "passed": passed, "total": total,
               "errors": sum(r["error"] is not None for r in results),
               "median_s": secs[len(secs) // 2] if secs else None, "max_s": secs[-1] if secs else None}
    return round(100 * passed / max(total, 1), 1), results, summary
