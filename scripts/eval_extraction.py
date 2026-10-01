"""Compare extraction quality across LLMs on a fixed, labelled set of memories.

    uv run python scripts/eval_extraction.py --chain ollama --ollama-model qwen2.5-coder:7b
    uv run python scripts/eval_extraction.py --chain ollama --ollama-model qwen3:8b
    uv run python scripts/eval_extraction.py --chain together            # needs TOGETHER_API_KEY in .env

Scores the parts recall depends on: how many events, what kind, which day, which entities,
and that facts / procedures / session notes land in the right layer. No DB writes.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.config  # noqa: E402,F401  loads .env
from src.memory import USER_TZ, local_date  # noqa: E402
from src.memory.extract import CAREER_TOPIC, extract  # noqa: E402

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


# (text, [(check name, predicate)])
CASES = [
    ("I applied to Stripe for a backend engineer role today through LinkedIn.", [
        ("1 applied", lambda x: kinds(x) == ["applied"]),
        ("date today", lambda x: days(x) == ["2026-10-01"]),
        ("entity stripe", lambda x: has(x, "stripe")),
    ]),
    ("Applied to Zomato and Swiggy for SDE-2 roles last Monday.", [
        ("2 applied (one per company)", lambda x: kinds(x) == ["applied", "applied"]),
        ("date last Monday", lambda x: days(x) and set(days(x)) == {"2026-09-28"}),
        ("entities both", lambda x: has(x, "zomato") and has(x, "swiggy")),
    ]),
    ("Yesterday I had the Razorpay technical round. DSA went great but I struggled with the system design question.", [
        ("1 interview", lambda x: kinds(x) == ["interview"]),
        ("date yesterday", lambda x: days(x) == ["2026-09-30"]),
        ("entity razorpay", lambda x: has(x, "razorpay")),
    ]),
    ("Got rejected by Stripe after the recruiter screen, they wanted more distributed systems experience.", [
        ("1 rejection", lambda x: kinds(x) == ["rejection"]),
        ("outcome rejected", lambda x: [e["outcome"] for e in x.episodes] == ["rejected"]),
        ("entity stripe", lambda x: has(x, "stripe")),
    ]),
    ("I prefer short, direct answers, and I use Arch Linux with Hyprland on my laptop.", [
        ("no events", lambda x: kinds(x) == []),
        ("≥1 preference fact", lambda x: any(f["kind"] == "preference" for f in x.facts)),
        ("not tagged job search", lambda x: not any(CAREER_TOPIC.lower() in [n.lower() for n in f["entities"]]
                                                     for f in x.facts)),
    ]),
    ("Moved to Bangalore last month for my new job at Example Labs.", [
        ("1 event", lambda x: len(x.episodes) == 1),
        ("date in September", lambda x: len(days(x)) == 1 and days(x)[0].startswith("2026-09")),
        ("fact about Bangalore", lambda x: any("bangalore" in f["text"].lower() for f in x.facts)),
    ]),
    ("Ran a 5k this morning in 28 minutes, feeling great.", [
        ("1 event", lambda x: len(x.episodes) == 1),
        ("date today", lambda x: days(x) == ["2026-10-01"]),
        ("not tagged job search", lambda x: not has(x, "job search")),
    ]),
    ("When I deploy CityFix I run the tests, build the docker image, push it to Railway, then check the health endpoint.", [
        ("procedure captured", lambda x: len(x.procedures) >= 1),
        ("≥4 steps", lambda x: bool(x.procedures) and len(x.procedures[0]["steps"]) >= 4),
        ("entity cityfix", lambda x: has(x, "cityfix")),
    ]),
    ("For this report: it needs to be a PDF and it's due Friday.", [
        ("no events", lambda x: kinds(x) == []),
        ("session notes", lambda x: len(x.session_notes) >= 1),
        ("mentions PDF", lambda x: any("pdf" in (n["key"] + n["value"]).lower() for n in x.session_notes)),
    ]),
    ("Had the final round at Acme on September 25th and got the offer today!", [
        ("interview + offer", lambda x: kinds(x) == ["interview", "offer"]),
        ("dates 25th and today", lambda x: days(x) == ["2026-09-25", "2026-10-01"]),
        ("entity acme", lambda x: has(x, "acme")),
    ]),
    ("What's the weather like today?", [
        ("nothing extracted", lambda x: not x.episodes and not x.facts and not x.procedures),
    ]),
    ("""[Claude Code session in project jobsearch]
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
    ("""[Claude Code session in project cityfix]
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
    ("My manager Priya asked me to lead the payments migration project starting next week.", [
        ("person priya", lambda x: has(x, "priya")),
        ("something stored", lambda x: bool(x.episodes or x.facts)),
    ]),
]


async def _case(text: str, checks, sem: asyncio.Semaphore):
    """Run one case -> (text, failed check names, latency or None, error or None)."""
    async with sem:
        t = time.monotonic()
        try:
            ex = await extract(text, NOW)
        except Exception as e:  # unusable output from every provider in the chain
            return text, [n for n, _ in checks], None, str(e)[:160]
        lat = time.monotonic() - t
    failed = []
    for name, pred in checks:
        try:
            ok = bool(pred(ex))
        except Exception:
            ok = False
        if not ok:
            failed.append(name)
    return text, failed, lat, None


async def run(label: str, repeats: int, concurrency: int) -> None:
    sem = asyncio.Semaphore(concurrency)  # reasoning models take 10-60s a call; don't run them one by one
    jobs = [(text, checks) for _ in range(repeats) for text, checks in CASES]
    total = sum(len(c) for _, c in jobs)
    passed = errors = 0
    latencies: list[float] = []
    failures: dict[str, int] = {}
    for done, fut in enumerate(asyncio.as_completed([_case(t, c, sem) for t, c in jobs]), 1):
        text, failed, lat, err = await fut
        n_checks = next(len(c) for t, c in CASES if t == text)
        passed += n_checks - len(failed)
        errors += err is not None
        if lat is not None:
            latencies.append(lat)
        for name in failed:
            key = f"{text[:40]}… / {name}"
            failures[key] = failures.get(key, 0) + 1
        # printed as they land: a killed run still shows everything finished so far
        status = f"ERROR {err}" if err else ("ok" if not failed else "fail: " + ", ".join(failed))
        print(f"[{done}/{len(jobs)}] {text[:40]!r}: {status}" + (f" ({lat:.1f}s)" if lat else ""), flush=True)
    lat = sorted(latencies)
    summary = (f"median {lat[len(lat) // 2]:.1f}s · max {lat[-1]:.1f}s" if lat else "no successful calls")
    print(f"\n=== {label}: {passed}/{total} checks ({100 * passed / max(total, 1):.0f}%) · errors {errors} · {summary}")
    for k, n in sorted(failures.items()):
        print(f"  ✗ {k}  ({n}/{repeats})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", required=True, help="e.g. ollama, together")
    ap.add_argument("--ollama-model")
    ap.add_argument("--together-model")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=5, help="parallel calls (use 1 for local Ollama)")
    a = ap.parse_args()
    os.environ["MEMORY_LLM_CHAIN"] = a.chain  # no fallback: measure exactly one model
    if a.ollama_model:
        os.environ["OLLAMA_LLM_MODEL"] = a.ollama_model
    if a.together_model:
        os.environ["MEMORY_MODEL_TOGETHER"] = a.together_model
    model = a.ollama_model if a.chain == "ollama" else (a.together_model or os.environ.get(
        f"MEMORY_MODEL_{a.chain.upper()}") or "default")
    asyncio.run(run(f"{a.chain}:{model}", a.repeats, a.concurrency))


if __name__ == "__main__":
    main()
