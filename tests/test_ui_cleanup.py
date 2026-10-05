import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select, text as sql

from src import settings_store as store
from src.memory import cleanup, entities as ent_mod
from src.memory.models import Entity, Episode, Reflection

LOCAL = {"host": "127.0.0.1:8001"}
T0 = datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _fresh_settings(monkeypatch):
    monkeypatch.setattr(store, "_cache", {})
    cleanup._cache.clear()
    yield
    cleanup._cache.clear()


def vec(i: int, j: int | None = None):
    v = [0.0] * 768
    v[i] = 1.0
    if j is not None:
        v[j] = 0.2
    return v


async def seed(session):
    for slug, kind, name in [
        ("company:redfox", "company", "Redfox"), ("company:redfox-cybersecurity", "company", "Redfox Cybersecurity"),
        ("company:redfox-recruiter", "company", "Redfox recruiter"),
        ("project:voiceagent", "project", "voiceAgent"), ("project:voice-agent", "project", "voice agent"),
        ("project:networkchainsopenclaw", "project", "networkchainsopenclaw"),
        ("project:networkchainopenclaw", "project", "NetworkChainOpenclaw"),
        ("project:networkchains", "project", "NetworkChains"),
        ("project:networkchains-downline-page", "project", "NetworkChains downline page"),
        ("feature:dark-mode", "feature", "Dark mode"), ("feature:networkchains", "feature", "NetworkChains"),
        ("person:me", "person", "Me"), ("topic:job-search", "topic", "Job search"),
    ]:
        session.add(Entity(slug=slug, kind=kind, name=name, aliases=["Redfox Security"] if slug == "company:redfox" else [],
                           dirty=False))
    session.add_all([
        Episode(id=1, occurred_at=T0, kind="interview", summary="Completed Redfox screening interview",
                entities=["company:redfox", "topic:job-search"], embedding=vec(0, 1)),
        Episode(id=2, occurred_at=T0 + timedelta(hours=1), kind="interview", summary="Completed Redfox screening call",
                outcome="take-home assigned", entities=["company:redfox", "person:me"], embedding=vec(0)),
        Episode(id=3, occurred_at=T0, kind="meeting", summary="Talked about voice agents",
                entities=["company:redfox-cybersecurity", "project:voiceagent", "person:me"], embedding=vec(5)),
        Episode(id=4, occurred_at=T0, kind="milestone", summary="Shipped networkchains",
                entities=["project:networkchains", "project:networkchains-downline-page", "feature:dark-mode"],
                embedding=vec(9)),
        Episode(id=5, occurred_at=T0 - timedelta(days=3), kind="milestone", summary="Shipped networkchains again",
                entities=["project:networkchains", "person:me"], embedding=vec(9)),
    ])
    session.add(Reflection(lesson="Redfox moves fast", entities=["company:redfox-cybersecurity", "company:redfox"],
                           evidence=[1, 2]))
    await session.commit()


async def add_fact(session, ents, **extra):
    fid = str(uuid.uuid4())
    await session.execute(sql("insert into mem0_test (id, payload) values (cast(:id as uuid), cast(:p as jsonb))"),
                          {"id": fid, "p": json.dumps({"data": "x", "user_id": "me", "entities": ents, **extra})})
    await session.commit()
    return fid


async def test_suggestions(session, mem0_store):
    await seed(session)
    groups = await cleanup.merge_groups(session)
    sets = [{i["slug"] for i in g["items"]} for g in groups]
    assert {"company:redfox", "company:redfox-cybersecurity", "company:redfox-recruiter"} in sets
    assert {"project:voiceagent", "project:voice-agent"} in sets
    assert any({"project:networkchainsopenclaw", "project:networkchainopenclaw"} <= s for s in sets)
    redfox = next(g for g in groups if "company:redfox" in {i["slug"] for i in g["items"]})
    assert redfox["keep"] == "company:redfox"  # most events

    dupes = await cleanup.dupe_groups(session)
    assert [[e["id"] for e in g["items"]] for g in dupes] == [[1, 2]]  # e:5 is another day, e:3 another kind

    kinds = {k["kind"]: k for k in await cleanup.kinds(session)}
    assert kinds["feature"]["to"] == "project" and not kinds["feature"]["canon"] and kinds["topic"]["canon"]

    once = {o["slug"]: o for o in await cleanup.once(session)}
    assert once["project:networkchains-downline-page"]["into"] == "project:networkchains"  # name contained
    assert once["feature:dark-mode"]["into"] == "project:networkchains"  # busiest co-occurring topic
    assert "project:networkchains" not in once

    assert (await cleanup.self_suggestion(session))["slug"] == "person:me"
    assert await cleanup.badge(session) == len(groups) + 1

    await cleanup.dismiss(session, redfox["key"])
    assert redfox["key"] not in [g["key"] for g in await cleanup.merge_groups(session)]


async def test_merge_endpoint(client, session):
    await seed(session)
    fid = await add_fact(session, ["company:redfox", "company:redfox-cybersecurity"], importance=4, kind="fact")
    await store.put(session, "recall", {"self_entity": "company:redfox", "pinned_cap": 0.3})

    r = await client.post("/api/cleanup/merge", headers=LOCAL,
                          json={"keep": "company:redfox-cybersecurity", "merge": ["company:redfox", "company:redfox-recruiter"]})
    assert r.status_code == 200, r.text
    assert r.json() == {"episodes": 2, "reflections": 1, "facts": 1, "merged": 2}

    session.expire_all()
    eps = {e.id: e.entities for e in (await session.execute(select(Episode))).scalars()}
    assert eps[1] == ["company:redfox-cybersecurity", "topic:job-search"]
    ref = (await session.execute(select(Reflection))).scalar_one()
    assert ref.entities == ["company:redfox-cybersecurity"]  # deduped
    payload = await session.scalar(sql("select payload from mem0_test where id = cast(:id as uuid)"), {"id": fid})
    assert payload["entities"] == ["company:redfox-cybersecurity"] and payload["importance"] == 4
    keep = (await session.execute(select(Entity).where(Entity.slug == "company:redfox-cybersecurity"))).scalar_one()
    assert keep.aliases == ["Redfox", "Redfox Security", "Redfox recruiter"] and keep.dirty
    assert await session.scalar(select(Entity.id).where(Entity.slug == "company:redfox")) is None
    assert store.get("recall") == {"self_entity": "company:redfox-cybersecurity", "pinned_cap": 0.3}

    r = await client.post("/api/cleanup/merge", headers=LOCAL, json={"keep": "company:redfox-cybersecurity", "merge": ["company:nope"]})
    assert r.status_code == 404 and "company:nope" in r.json()["detail"]
    r = await client.post("/api/cleanup/merge", headers=LOCAL, json={"keep": "project:voiceagent", "merge": ["project:voiceagent"]})
    assert r.status_code == 422


async def test_dedupe_endpoint(client, session):
    await seed(session)
    r = await client.post("/api/cleanup/dedupe", headers=LOCAL, json={"keep": 1, "remove": [1, 2]})
    assert r.status_code == 200, r.text
    session.expire_all()
    kept = await session.get(Episode, 1)
    assert kept.outcome == "take-home assigned"
    assert kept.entities == ["company:redfox", "topic:job-search", "person:me"]
    assert await session.get(Episode, 2) is None
    assert (await session.execute(select(Reflection))).scalar_one().evidence == [1]
    me = (await session.execute(select(Entity).where(Entity.slug == "person:me"))).scalar_one()
    assert me.dirty
    r = await client.post("/api/cleanup/dedupe", headers=LOCAL, json={"keep": 1, "remove": [2]})
    assert r.status_code == 404


async def test_kinds_and_resolve(client, session, monkeypatch):
    await seed(session)
    r = await client.post("/api/cleanup/kinds", headers=LOCAL, json={"map": {"feature": "project"}})
    assert r.status_code == 200, r.text
    assert r.json() == {"moved": 1, "merged": 1, "kinds": 1}
    session.expire_all()
    slugs = set((await session.execute(select(Entity.slug))).scalars())
    assert "project:dark-mode" in slugs and not {"feature:dark-mode", "feature:networkchains"} & slugs
    assert "project:dark-mode" in (await session.get(Episode, 4)).entities
    assert store.get("kind_map") == {"feature": "project"}

    r = await client.post("/api/cleanup/kinds", headers=LOCAL, json={"map": {"app": "gadget"}})
    assert r.status_code == 422

    async def no_embed(_):
        return [0.0] * 768
    monkeypatch.setattr(ent_mod, "embed", no_embed)
    out = await ent_mod.resolve(session, [{"name": "Search bar", "kind": "feature"}, {"name": "Pixel", "kind": "gizmo"}])
    assert out == {"Search bar": "project:search-bar", "Pixel": "topic:pixel"}


async def test_fold_endpoint(client, session):
    await seed(session)
    r = await client.post("/api/cleanup/fold", headers=LOCAL, json={"items": [
        {"slug": "project:networkchains-downline-page", "into": "project:networkchains"},
        {"slug": "feature:dark-mode", "into": "project:networkchains-downline-page"},  # parent folded above
    ]})
    assert r.status_code == 200, r.text
    assert r.json()["merged"] == 2
    session.expire_all()
    assert (await session.get(Episode, 4)).entities == ["project:networkchains"]
    nc = (await session.execute(select(Entity).where(Entity.slug == "project:networkchains"))).scalar_one()
    assert set(nc.aliases) == {"NetworkChains downline page", "Dark mode"}


async def test_page_renders(client, session):
    await seed(session)
    r = await client.get("/cleanup", headers=LOCAL)
    assert r.status_code == 200
    html = r.text
    assert "Same thing, two names" in html and "Redfox Cybersecurity" in html and "Is this you?" in html
    assert 'data-api="POST /api/cleanup/merge"' in html and 'data-api="POST /api/cleanup/dedupe"' in html
    assert "kinds-form" in html and 'data-api="POST /api/cleanup/fold"' in html
