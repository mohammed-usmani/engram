import pytest
from datetime import datetime, timezone

from sqlalchemy import select

from src.memory import entities, extract
from src.memory.models import Entity



@pytest.fixture(autouse=True)
def _placeholder_text_is_grounded(request, monkeypatch):
    """These tests feed placeholder text ("...") to check field clean-up; grounding has its own test."""
    if "recaps" not in request.node.name:
        from src.memory import extract as _ex
        monkeypatch.setattr(_ex, "_grounded", lambda *a, **k: True)

def test_redact():
    text = ("key sk-ant-api03-abcdefghijklmnopqrstuv and ghp_abcdefghijklmnopqrstuvwxyz0123 "
            "header Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.abc.def card 4111 1111 1111 1111 ok")
    out = extract.redact(text)
    for secret in ("sk-ant-api03", "ghp_abc", "eyJhbGci", "4111 1111"):
        assert secret not in out
    assert out.endswith("ok")


def test_slugify():
    assert entities.slugify("company", "Acme Corp.") == "company:acme-corp"
    assert entities.slugify("topic", "Job Search") == "topic:job-search"


async def test_extract_coerces_bad_fields(monkeypatch):
    fixture = {
        "entities": [{"name": "Acme", "kind": "company"}, {"kind": "company"}],
        "episodes": [
            {"kind": "interview", "summary": "System design round at Acme went badly",
             "when": "2026-09-29", "entities": ["Acme"], "outcome": "pending",
             "importance": "high", "sentiment": -0.6},
            {"kind": "applied"},  # no summary -> dropped
        ],
        "facts": [{"text": "Targets backend roles", "kind": "preference", "entities": [], "importance": 4},
                  {"kind": "fact"}],
        "procedures": [{"name": "apply", "steps": ["cv send", "log it"]}],
        "session_notes": [{"key": "deadline", "value": "Friday"}],
    }

    async def fake(prompt, system="", task="default"):
        assert "2026-10-01" in prompt  # reference date passed through
        return fixture

    monkeypatch.setattr(extract, "complete_json", fake)
    ex = await extract.extract("...", datetime(2026, 10, 1, 9, tzinfo=timezone.utc))
    assert [e["name"] for e in ex.entities] == ["Acme", "Job search"]  # career topic auto-added
    assert len(ex.episodes) == 1
    ep = ex.episodes[0]
    assert ep["importance"] == 3 and ep["occurred_at"].date().isoformat() == "2026-09-29"
    assert [f["text"] for f in ex.facts] == ["Targets backend roles"]
    assert ex.procedures[0]["steps"] == ["cv send", "log it"]
    assert ex.session_notes == [{"key": "deadline", "value": "Friday"}]


async def test_resolve_aliases(session, monkeypatch):
    async def fake_embed(text):
        return [0.0] * 768
    monkeypatch.setattr(entities, "embed", fake_embed)
    m1 = await entities.resolve(session, [{"name": "Acme Corp", "kind": "company"}])
    session.add(Entity(slug="topic:job-search", kind="topic", name="Job search", aliases=["job hunt", "applying"]))
    await session.commit()
    m2 = await entities.resolve(session, [{"name": "acme corp", "kind": "company"},
                                          {"name": "job hunt", "kind": "topic"}])
    assert m1["Acme Corp"] == m2["acme corp"] == "company:acme-corp"
    assert m2["job hunt"] == "topic:job-search"
    assert len((await session.execute(select(Entity))).scalars().all()) == 2
    assert await entities.match_text(session, "I'm applying for jobs at Acme Corp") == ["company:acme-corp", "topic:job-search"]


async def test_extract_normalizes_kinds_and_career_topic(monkeypatch):
    async def fake(prompt, system="", task="default"):
        return {"entities": [{"name": "Stripe", "kind": "company"}],
                "episodes": [{"kind": "application", "summary": "Applied to Stripe", "entities": ["Stripe"]},
                             {"kind": "technical round", "summary": "Razorpay tech round", "entities": []},
                             {"kind": "gardening", "summary": "Planted basil", "entities": []}],
                "facts": [], "procedures": [], "session_notes": []}
    monkeypatch.setattr(extract, "complete_json", fake)
    ex = await extract.extract("...", datetime(2026, 10, 1, 9, tzinfo=timezone.utc))
    assert [e["kind"] for e in ex.episodes] == ["applied", "interview", "other"]
    assert "Job search" in ex.episodes[0]["entities"] and "Job search" in ex.episodes[1]["entities"]
    assert "Job search" not in ex.episodes[2]["entities"]
    assert {"name": "Job search", "kind": "topic"} in ex.entities


def test_local_dates():
    from src.memory import local_date
    # 19:30 UTC on 30 Sep is 01:00 on 1 Oct in IST
    assert local_date(datetime(2026, 9, 30, 19, 30, tzinfo=timezone.utc)) == "2026-10-01"


async def test_outcome_normalization(monkeypatch):
    async def fake(prompt, system="", task="default"):
        return {"episodes": [{"kind": "rejection", "summary": "Stripe said no", "outcome": "pending"},
                             {"kind": "applied", "summary": "Applied to X", "outcome": "Pending"},
                             {"kind": "interview", "summary": "Y onsite", "outcome": "passed"}]}
    monkeypatch.setattr(extract, "complete_json", fake)
    ex = await extract.extract("...", datetime(2026, 10, 1, tzinfo=timezone.utc))
    assert [e["outcome"] for e in ex.episodes] == ["rejected", None, "passed"]


async def test_multi_company_career_event_is_split(monkeypatch):
    async def fake(prompt, system="", task="default"):
        assert "one episode per company" in prompt
        return {"entities": [{"name": "Zomato", "kind": "company"}, {"name": "Swiggy", "kind": "company"},
                             {"name": "SDE-2", "kind": "topic"}],
                "episodes": [{"kind": "applied", "summary": "Applied to Zomato and Swiggy for SDE-2 roles",
                              "entities": ["Zomato", "Swiggy", "SDE-2"], "when": "2026-09-28"}]}
    monkeypatch.setattr(extract, "complete_json", fake)
    ex = await extract.extract("...", datetime(2026, 10, 1, 9, tzinfo=timezone.utc))
    assert [e["kind"] for e in ex.episodes] == ["applied", "applied"]
    assert [sorted(e["entities"]) for e in ex.episodes] == [
        ["Job search", "SDE-2", "Zomato"], ["Job search", "SDE-2", "Swiggy"]]
    assert all(e["occurred_at"].date().isoformat() == "2026-09-28" for e in ex.episodes)


async def test_one_off_todos_are_not_procedures(monkeypatch):
    async def fake(prompt, system="", task="default"):
        assert "repeatable" in prompt and "never \"the app\"" in prompt
        return {"procedures": [
            {"name": "Deciding to merge the branch into main", "steps": ["Decide whether to merge feat/x"]},
            {"name": "deploy cityfix", "steps": ["run tests", "build image", "push", "verify"]}]}
    monkeypatch.setattr(extract, "complete_json", fake)
    ex = await extract.extract("...", datetime(2026, 10, 1, tzinfo=timezone.utc))
    assert [p["name"] for p in ex.procedures] == ["deploy cityfix"]


def test_recaps_cannot_invent_applications_or_dates():
    """Seen live: a history recap turned every company into "Applied to X", each with a made-up date."""
    from datetime import datetime, timezone
    from src.memory.extract import parse_extraction
    now = datetime(2026, 10, 6, 12, tzinfo=timezone.utc)
    text = ("Acme — Backend Engineer. Applied Oct 6 2026.\n\n"
            "Globex — AI Engineer. Strong target, tailored resume created.\n\n"
            "Initech — screening around Oct 5 2026 went well.\n\n"
            "Umbrella — interview, details unknown.")
    data = {"entities": [{"name": n, "kind": "company"} for n in ("Acme", "Globex", "Initech", "Umbrella")],
            "episodes": [
                {"kind": "applied", "summary": "Applied to Acme", "when": "2026-10-06", "entities": ["Acme"]},
                {"kind": "applied", "summary": "Applied to Globex", "when": "2026-10-04", "entities": ["Globex"]},
                {"kind": "interview", "summary": "Initech screening", "when": "2026-10-05", "entities": ["Initech"]},
                {"kind": "interview", "summary": "Umbrella interview", "when": "2026-10-02", "entities": ["Umbrella"]}]}
    kept = [e["summary"] for e in parse_extraction(data, now, text).episodes]
    assert kept == ["Applied to Acme", "Initech screening"]
    # without the source text (old callers) nothing is filtered
    assert len(parse_extraction(data, now).episodes) == 4
