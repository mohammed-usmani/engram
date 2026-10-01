from pathlib import Path
from sqlalchemy import select
from src.seed.loader import run_seed
from src.models import ContextDocument, Source, DocType


def _write(tmp: Path, sub: str, name: str, body: str) -> None:
    d = tmp / sub
    d.mkdir(exist_ok=True)
    (d / name).write_text(body)


async def test_seeds_empty_db(session, tmp_path):
    _write(tmp_path, "skills", "python.txt", "Type: Skill\n\nSkill Name: Python\n\nSummary:\nHi.\n")
    _write(tmp_path, "projects", "x.txt", "Type: Project\n\nTitle: X\n\nSummary:\nHi.\n")
    summary = await run_seed(session, tmp_path)
    await session.commit()
    assert summary["inserted"] == 2
    assert summary["updated"] == 0
    assert summary["errors"] == 0

    rows = (await session.execute(select(ContextDocument))).scalars().all()
    slugs = {r.slug for r in rows}
    assert slugs == {"skill/python", "project/x"}


async def test_reseed_is_idempotent_and_updates_file_rows(session, tmp_path):
    _write(tmp_path, "skills", "python.txt", "Type: Skill\n\nSkill Name: Python\n\nSummary:\nOld.\n")
    await run_seed(session, tmp_path)
    await session.commit()

    (tmp_path / "skills" / "python.txt").write_text("Type: Skill\n\nSkill Name: Python\n\nSummary:\nNew.\n")
    summary = await run_seed(session, tmp_path)
    await session.commit()
    assert summary["inserted"] == 0
    assert summary["updated"] == 1
    row = (await session.execute(select(ContextDocument).where(ContextDocument.slug == "skill/python"))).scalar_one()
    assert "New." in row.sections.get("Summary", "")
    assert row.source == Source.FILE


async def test_reseed_preserves_manual_rows(session, tmp_path):
    _write(tmp_path, "skills", "python.txt", "Type: Skill\n\nSkill Name: Python\n\nSummary:\nFile.\n")
    session.add(ContextDocument(
        type=DocType.SKILL, slug="skill/handwritten", title="Handwritten",
        tags=["skill"], content="manual content", sections={}, source=Source.MANUAL, doc_metadata={},
    ))
    await session.commit()

    await run_seed(session, tmp_path)
    await session.commit()

    manual = (await session.execute(select(ContextDocument).where(ContextDocument.slug == "skill/handwritten"))).scalar_one()
    assert manual.content == "manual content"
    assert manual.source == Source.MANUAL
