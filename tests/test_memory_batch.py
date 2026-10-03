import json
from datetime import datetime, timezone

import httpx
from sqlalchemy import select

from src.memory import batch, ingest
from src.memory.models import Episode, IngestJob

WHEN = datetime(2026, 10, 3, 10, tzinfo=timezone.utc)
GOOD = {"entities": [{"name": "Acme", "kind": "company"}],
        "episodes": [{"kind": "interview", "summary": "Had the Acme onsite", "when": "2026-10-03",
                      "entities": ["Acme"], "importance": 4}],
        "facts": [], "procedures": [], "session_notes": []}


def fake_together(monkeypatch):
    """Answers like Together's batch API: the first job succeeds, the second lands in the error file."""
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        path = req.url.path
        if path == "/v1/files/upload":
            seen["lines"] = [json.loads(line) for line in req.content.decode().splitlines()
                             if line.startswith('{"custom_id"')]
            return httpx.Response(200, json={"id": "file-in"})
        if path == "/v1/batches":
            return httpx.Response(201, json={"job": {"id": "b1", "status": "VALIDATING"}})
        if path == "/v1/batches/b1":
            return httpx.Response(200, json={"job": {"id": "b1", "status": "COMPLETED",
                                                     "output_file_id": "file-out", "error_file_id": "file-err"}})
        if path == "/v1/files/file-out/content":
            return httpx.Response(200, text=json.dumps({"custom_id": seen["first"], "response": {
                "status_code": 200, "body": {"choices": [{"message": {"content": json.dumps(GOOD)}}]}}}) + "\n")
        if path == "/v1/files/file-err/content":
            return httpx.Response(200, text=json.dumps({"custom_id": seen["second"],
                                                        "error": {"message": "overloaded"}}) + "\n")
        return httpx.Response(404)

    monkeypatch.setenv("TOGETHER_API_KEY", "t")
    monkeypatch.setattr(batch, "_client", lambda: httpx.AsyncClient(
        base_url=batch.API, transport=httpx.MockTransport(handler)))
    return seen


async def _two_jobs(session, seen):
    a, _ = await ingest.enqueue(session, "I had the Acme onsite", agent="t", occurred_at=WHEN)
    b, _ = await ingest.enqueue(session, "another chunk", agent="t", occurred_at=WHEN)
    seen["first"], seen["second"] = str(a), str(b)
    return a, b


async def test_batch_mode_sends_the_queue_and_saves_results(session, mem0_store, monkeypatch):
    seen = fake_together(monkeypatch)
    a, b = await _two_jobs(session, seen)

    await batch.set_mode(session, True)
    assert await ingest.process_pending(session) == 0  # live workers stand aside
    await batch.tick(session)
    assert [line["custom_id"] for line in seen["lines"]] == [str(a), str(b)]
    assert seen["lines"][0]["body"]["model"] == batch.model()
    assert (await session.get(IngestJob, a)).status == "batched"

    await batch.tick(session)  # finished: saved exactly like a live extraction
    assert (await session.get(IngestJob, a)).status == "done"
    assert (await session.execute(select(Episode.summary))).scalars().all() == ["Had the Acme onsite"]
    redo = await session.get(IngestJob, b)  # the failed one counts an attempt and is sent again
    assert redo.attempts == 1 and redo.status == "batched" and seen["lines"][0]["custom_id"] == str(b)


async def test_switching_off_still_collects_sent_batches(session, mem0_store, monkeypatch):
    seen = fake_together(monkeypatch)
    a, b = await _two_jobs(session, seen)
    await batch.set_mode(session, True)
    await batch.tick(session)
    await batch.set_mode(session, False)
    await batch.tick(session)
    assert (await session.get(IngestJob, a)).status == "done"
    assert (await session.get(IngestJob, b)).status == "pending"  # back to the live workers
