from sqlalchemy import select

from src.memory import backfill
from src.memory.models import Episode

MD = """# Applications

| Date | Company | Role | Via | CV | Status | Last contact | Next action |
|------|---------|------|-----|----|--------|--------------|-------------|
| 2026-09-20 | Acme Corp | Backend Engineer | LinkedIn | bd90ae7 | onsite | 2026-09-26 | wait |
| 2026-09-18 | Beta Labs | AI Engineer | referral | bd90ae7 | rejected | | |
| | | | | | | | |
"""


async def test_sync_applications_idempotent_and_updates(session, tmp_path, monkeypatch):
    async def fake_embed(text):
        return [0.1] * 768
    monkeypatch.setattr(backfill, "embed", fake_embed)
    f = tmp_path / "applications.md"
    f.write_text(MD)
    assert await backfill.sync_applications(session, f) == {"added": 2, "updated": 0}
    assert await backfill.sync_applications(session, f) == {"added": 0, "updated": 0}

    f.write_text(MD.replace("| onsite |", "| offer |"))
    assert await backfill.sync_applications(session, f) == {"added": 0, "updated": 1}
    eps = (await session.execute(select(Episode).order_by(Episode.occurred_at))).scalars().all()
    assert [(e.kind, e.outcome) for e in eps] == [("applied", "rejected"), ("applied", "offer")]
    assert set(eps[1].entities) == {"company:acme-corp", "topic:job-search"}
    assert "Backend Engineer" in eps[1].summary
