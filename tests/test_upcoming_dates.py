"""Dates inside documents (Expires:, Renews:, Due:, Birthday:...) surface as 'Coming up'."""
from datetime import date

import pytest

from tests.test_mcp_tools import _call, mcp_db  # noqa: F401  (fixture)

import src.mcp_server as srv
from src import dates
from src.privacy import REQUEST_IS_REMOTE

TODAY = date(2026, 10, 5)


def test_parse_formats():
    assert dates.parse_date("2027-03-14") == (date(2027, 3, 14), False)
    assert dates.parse_date("14 Mar 2027") == (date(2027, 3, 14), False)
    assert dates.parse_date("March 14, 2027") == (date(2027, 3, 14), False)
    assert dates.parse_date("14/03/2027") == (date(2027, 3, 14), False)   # day first
    assert dates.parse_date("10th Oct") == (date(2000, 10, 10), True)     # no year: yearly
    assert dates.parse_date("someday") == (None, False)


async def _doc(slug, content, privacy="normal", type_="note"):
    return await _call(srv.save_document)(slug=slug, agent="claude-code", title=slug.split("/")[1].title(),
                                          content=content, type=type_, privacy=privacy)


async def test_upcoming_window_yearly_and_privacy(mcp_db):
    await _doc("travel/passport", "Number: Z1234567\nExpires: 2026-11-28\n\nNotes:\nRenew 3 months early.\n",
               privacy="private", type_="travel")
    await _doc("person/sam", "Relation: friend\nBirthday: 10 Oct 1976\n", type_="person")
    await _doc("home/car-insurance", "Insurer: ACKO\nRenews: 14/03/2027\n", type_="home")
    await _doc("finance/fd", "Bank: SBI\nMatures: 2026-10-20\n", privacy="local-only", type_="finance")
    await _doc("note/broken", "Due: sometime soon\n")
    async with srv.AsyncSessionLocal() as s:
        items, unreadable = await dates.upcoming(s, days=60, today=TODAY)
    labels = {i["slug"]: i for i in items}
    assert labels["person/sam"]["date"] == "2026-10-10" and labels["person/sam"]["turns"] == 50
    assert labels["travel/passport"]["label"] == "expires" and labels["travel/passport"]["days_left"] == 54
    assert "home/car-insurance" not in labels                 # outside 60 days
    assert "Z1234567" not in str(items)                       # private: reminder only, never content
    assert any(u["slug"] == "note/broken" for u in unreadable)
    assert [i["slug"] for i in items] == sorted(labels, key=lambda k: labels[k]["date"])


async def test_local_only_dates_hidden_remotely(mcp_db):
    await _doc("finance/fd", "Bank: SBI\nDue: 2026-10-20\n", privacy="local-only", type_="finance")
    tok = REQUEST_IS_REMOTE.set(True)
    try:
        async with srv.AsyncSessionLocal() as s:
            items, _ = await dates.upcoming(s, days=60, today=TODAY)
    finally:
        REQUEST_IS_REMOTE.reset(tok)
    assert all(i["slug"] != "finance/fd" for i in items)


async def test_upcoming_tool_and_recall_section(mcp_db, monkeypatch):
    monkeypatch.setattr(dates, "_today", lambda: TODAY)
    await _doc("person/sam", "Birthday: 10 Oct 1976\n", type_="person")
    out = await _call(srv.upcoming)(days=30)
    assert out["items"][0]["slug"] == "person/sam"
    line = dates.format_line(out["items"][0])
    assert "birthday" in line and "in 5 days" in line and "turns 50" in line


async def test_upcoming_bad_days_and_rest(mcp_db, client):
    out = await _call(srv.upcoming)(days=0)
    assert "days" in out["error"] and "fix" in out
    r = await client.get("/api/upcoming?days=30", headers={"host": "127.0.0.1:8001"})
    assert r.status_code == 200 and "items" in r.json()


def test_body_dates_weekday_conflict_and_add_field():
    meta = {"Type": "Interview record"}
    found = dates.body_dates("Type: Interview record\n\nTake-home:\n- Deadline: Wednesday 8 Oct 2026.\n", meta)
    assert found[0]["date"] == "2026-10-08" and found[0]["weekday_conflict"] == "Thursday" and found[0]["line"] == 4
    assert dates.body_dates("Type: x\nDue: 2026-10-07\n\nNotes:\n- Deadline: 7 Oct 2026\n", {"Type": "x", "Due": "2026-10-07"}) == []
    out = dates.add_field("Type: Interview record\nOutcome: Advanced\n\nNotes:\nhi\n", {"Type": 1, "Outcome": 1}, "Due", "2026-10-07")
    assert out.splitlines()[:3] == ["Type: Interview record", "Outcome: Advanced", "Due: 2026-10-07"]
