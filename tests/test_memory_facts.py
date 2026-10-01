"""Hits the test DB and local Ollama embeddings (nomic-embed-text)."""
from src.memory import facts


async def test_add_search_supersede(mem0_store):
    a = await facts.add_fact("Prefers NestJS over Express for production APIs", kind="preference",
                             entities=["tech:nestjs"], importance=4, agent="claude-code")
    await facts.add_fact("Deploy CityFix: run tests, build image, push, verify", kind="procedure",
                         entities=["project:cityfix"])

    hits = await facts.search_facts("which backend framework do I like", kinds=["preference"])
    assert hits and hits[0]["id"] == a
    assert hits[0]["metadata"]["entities"] == ["tech:nestjs"]
    assert all(h["metadata"]["kind"] == "preference" for h in hits)

    b = await facts.add_fact("Now prefers FastAPI over NestJS", kind="preference", entities=["tech:nestjs"])
    await facts.supersede(a, b)
    ids = [h["id"] for h in await facts.search_facts("backend framework preference")]
    assert a not in ids and b in ids
    old = await facts.get_fact(a)
    assert old["metadata"]["superseded_by"] == b

    await facts.delete_fact(b)
    assert await facts.get_fact(b) is None
    assert {f["id"] for f in await facts.all_facts(kinds=["procedure"])}
