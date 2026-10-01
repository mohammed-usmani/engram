from pathlib import Path
import pytest
from src.seed.parser import parse_file
from src.models import DocType


def _make(tmp_path: Path, category: str, filename: str, body: str) -> Path:
    d = tmp_path / category
    d.mkdir()
    p = d / filename
    p.write_text(body)
    return p


def test_parses_skill_file(tmp_path):
    body = (
        "Type: Skill\n\n"
        "Skill Name: Python\n\n"
        "Category: Languages\n\n"
        "Level: Intermediate\n\n"
        "Summary:\nExperienced in Python for backend work.\n\n"
        "Key Concepts:\n- OOP\n- asyncio\n"
    )
    p = _make(tmp_path, "skills", "python.txt", body)
    parsed = parse_file(p)
    assert parsed["type"] == DocType.SKILL
    assert parsed["slug"] == "skill/python"
    assert parsed["title"] == "Python"
    assert "skill" in parsed["tags"] and "languages" in parsed["tags"] and "intermediate" in parsed["tags"]
    assert "Summary" in parsed["sections"]
    assert "Experienced in Python" in parsed["sections"]["Summary"]
    assert "Key Concepts" in parsed["sections"]
    assert parsed["metadata"].get("Category") == "Languages"
    assert parsed["metadata"].get("Level") == "Intermediate"
    assert parsed["content"] == body


def test_parses_experience_file(tmp_path):
    body = (
        "Type: Experience\n\n"
        "Title: Freelance Full-Stack Developer\n\n"
        "Situation:\nManaged end-to-end development.\n\n"
        "Task:\nDesign, build, deploy.\n\n"
        "Action:\n- Built X\n- Shipped Y\n\n"
        "Result:\n- Delivered scalable systems.\n"
    )
    p = _make(tmp_path, "work", "freelance.txt", body)
    parsed = parse_file(p)
    assert parsed["type"] == DocType.EXPERIENCE
    assert parsed["slug"] == "experience/freelance"
    assert parsed["title"] == "Freelance Full-Stack Developer"
    assert set(parsed["sections"].keys()) >= {"Situation", "Task", "Action", "Result"}


def test_parses_education_file(tmp_path):
    body = (
        "Type: Education\n\n"
        "Title: B.E. in CS\n\n"
        "Situation:\nEnrolled in undergrad.\n\n"
        "Result:\nCGPA 8.0.\n"
    )
    p = _make(tmp_path, "education", "btech.txt", body)
    parsed = parse_file(p)
    assert parsed["type"] == DocType.EDUCATION
    assert parsed["slug"] == "education/btech"
    assert parsed["title"] == "B.E. in CS"


def test_resume_type(tmp_path):
    body = "MOHAMMED USMANI\nEmail: x@y.z\n"
    p = _make(tmp_path, "resume", "master_resume.txt", body)
    parsed = parse_file(p)
    assert parsed["type"] == DocType.RESUME
    assert parsed["slug"] == "resume/master_resume"
    assert parsed["title"]


def test_malformed_file_does_not_crash(tmp_path):
    body = "just random text with no headers\nmore text"
    p = _make(tmp_path, "skills", "weird.txt", body)
    parsed = parse_file(p)
    assert parsed["type"] == DocType.SKILL
    assert parsed["content"] == body
    assert isinstance(parsed["sections"], dict)


def test_unknown_directory_raises(tmp_path):
    p = _make(tmp_path, "unknown_kind", "x.txt", "Type: Foo\n")
    with pytest.raises(ValueError):
        parse_file(p)
