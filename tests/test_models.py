import pytest
from sqlalchemy import select
from src.models import ContextDocument, DocType, Source


async def test_insert_and_query_document(session):
    doc = ContextDocument(
        type=DocType.SKILL,
        slug="skill/python",
        title="Python",
        tags=["skill", "languages"],
        content="Type: Skill\n\nSkill Name: Python\n...",
        sections={"Summary": "Experienced in Python."},
        source=Source.FILE,
        doc_metadata={"Category": "Languages", "Level": "Intermediate"},
    )
    session.add(doc)
    await session.commit()

    result = await session.execute(
        select(ContextDocument).where(ContextDocument.slug == "skill/python")
    )
    row = result.scalar_one()
    assert row.title == "Python"
    assert row.tags == ["skill", "languages"]
    assert row.sections == {"Summary": "Experienced in Python."}
    assert row.source == Source.FILE
    assert row.type == DocType.SKILL


async def test_slug_unique(session):
    session.add(ContextDocument(
        type=DocType.SKILL, slug="skill/a", title="A",
        tags=[], content="x", sections={}, source=Source.FILE, doc_metadata={},
    ))
    await session.commit()
    session.add(ContextDocument(
        type=DocType.SKILL, slug="skill/a", title="A",
        tags=[], content="x", sections={}, source=Source.FILE, doc_metadata={},
    ))
    with pytest.raises(Exception):
        await session.commit()
