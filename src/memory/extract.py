"""Turn free text from any agent into typed memory candidates with one LLM call."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

from src.memory import USER_TZ
from src.memory.llm import complete_json

_SECRETS = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
    re.compile(r"\b(sk|pk|rk)-[A-Za-z0-9_\-]{16,}"),          # OpenAI/Anthropic/Stripe style
    re.compile(r"\b(ghp|gho|ghs|ghu|github_pat)_[A-Za-z0-9_]{20,}"),
    re.compile(r"\b(gsk|xai|AIza)[A-Za-z0-9_\-]{20,}"),         # Groq, xAI, Google
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),                        # AWS access key id
    re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),              # Slack
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{12,}"),
    re.compile(r"(?i)[\"']?\b(password|passwd|secret|api[_-]?key|token)\b[\"']?\s*[:=]\s*[\"']?[^\s\"',}]+"),
    re.compile(r"(?i)\b(password|passcode|passphrase|pin)\s+(is|was)\s+\S+"),
]
_URL_CREDS = re.compile(r"(\b[a-z][a-z0-9+.-]*://[^\s:/@]+:)[^\s@/]+@")
_DIGITS = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")


def _luhn(num: str) -> bool:
    ds = [int(c) for c in num if c.isdigit()][::-1]
    return sum(d if i % 2 == 0 else (d * 2 - 9 if d > 4 else d * 2) for i, d in enumerate(ds)) % 10 == 0


def redact(text: str) -> str:
    for rx in _SECRETS:
        text = rx.sub("[REDACTED]", text)
    text = _URL_CREDS.sub(r"\1[REDACTED]@", text)
    return _DIGITS.sub(lambda m: "[REDACTED]" if _luhn(m.group()) else m.group(), text)  # cards, not timestamps


SYSTEM = "You extract durable personal memory about the user from text an AI assistant saw. Output JSON only."

PROMPT = """Reference time (now): {now}
Known entities (reuse these exact names when they apply): {known}

Extract memory about THE USER from the text below. Skip chit-chat, questions without answers,
and anything only relevant to this one moment unless it is a session note.

Return JSON with these keys (use [] when nothing applies):
- "entities": [{{"name": str, "kind": "company|person|project|topic|place|tool|org"}}]
  Always include the broad ongoing life area as a topic (e.g. "Job search", "Health",
  "Fitness", "Finances", "Learning Rust") so related events group together.
- "episodes": things that HAPPENED in the user's work or life — one episode per company / person / item involved.
  Outcomes count (shipped a release, fixed a production bug, applied, decided); the coding session itself,
  tool/agent operations, builds, emulator runs and "we discussed X" do NOT.
  "applied to Zomato and Swiggy" is TWO "applied" episodes, one for each company.
  A RECAP of earlier history ("history so far", "consolidated", "previously", a status summary) is not new:
  make an episode for a past event only if the text gives its own date, and use THAT date, never the reference
  time. Undated past events in a recap are skipped (they are already in memory).
  "applied" only when the text says the application was submitted. Researching a company, tailoring a resume
  for it, rating the fit or calling it a target is NOT an application.
  [{{"kind": "applied|interview|offer|rejection|decision|meeting|milestone|incident|task_done|
    purchase|travel|learning|conversation|other", "summary": str (one sentence, past tense),
    "when": ISO date or datetime (resolve "yesterday", "last Friday" against now),
    "entities": [names], "outcome": str|null, "sentiment": -1..1|null, "importance": 1-5}}]
- "facts": durable truths about the user, their preferences, setup, people, goals — NOT events.
  Each fact must stand alone months later: name the project, product or person, never "the app" / "this repo"
  (anything with a date or "yesterday/today" is an episode, not a fact):
  [{{"text": str (standalone sentence), "kind": "fact|preference", "entities": [names], "importance": 1-5}}]
- "procedures": repeatable workflows the user does again and again, with 2+ steps
  (how they deploy, apply for jobs, release an app): [{{"name": str, "steps": [str], "entities": [names]}}]
  One-off to-dos and next steps from this conversation are NOT procedures — put them in "session_notes".
- "session_notes": temporary constraints for the current task (deadlines, formats): [{{"key": str, "value": str}}]

Importance: 5 = life-changing/identity, 4 = major, 3 = normal, 2 = minor, 1 = trivia.

Example — text: "Had my Acme onsite yesterday, system design went badly. I prefer remote roles."
{{"entities": [{{"name": "Acme", "kind": "company"}}, {{"name": "Job search", "kind": "topic"}}],
 "episodes": [{{"kind": "interview", "summary": "Had Acme onsite; system design round went badly",
   "when": "<yesterday's date>", "entities": ["Acme", "Job search"], "outcome": "pending", "sentiment": -0.5, "importance": 4}}],
 "facts": [{{"text": "Prefers remote roles", "kind": "preference", "entities": ["Job search"], "importance": 3}}],
 "procedures": [], "session_notes": []}}

Example — text: "Sent applications to Cred and Groww this morning."
{{"entities": [{{"name": "Cred", "kind": "company"}}, {{"name": "Groww", "kind": "company"}}, {{"name": "Job search", "kind": "topic"}}],
 "episodes": [{{"kind": "applied", "summary": "Applied to Cred", "when": "<today>", "entities": ["Cred", "Job search"], "importance": 3}},
              {{"kind": "applied", "summary": "Applied to Groww", "when": "<today>", "entities": ["Groww", "Job search"], "importance": 3}}],
 "facts": [], "procedures": [], "session_notes": []}}

TEXT:
{text}"""


@dataclass
class Extraction:
    entities: list[dict] = field(default_factory=list)
    episodes: list[dict] = field(default_factory=list)
    facts: list[dict] = field(default_factory=list)
    procedures: list[dict] = field(default_factory=list)
    session_notes: list[dict] = field(default_factory=list)


EPISODE_KINDS = {"applied", "interview", "offer", "rejection", "decision", "meeting", "milestone", "incident",
                 "task_done", "purchase", "travel", "learning", "conversation", "session_note", "other"}
_KIND_SYNONYMS = {
    "application": "applied", "apply": "applied", "job_application": "applied",
    "technical_round": "interview", "tech_round": "interview", "screen": "interview", "phone_screen": "interview",
    "onsite": "interview", "recruiter_screen": "interview", "interview_round": "interview",
    "job_offer": "offer", "rejected": "rejection", "reject": "rejection",
    "completed": "task_done", "done": "task_done", "task": "task_done", "achievement": "milestone",
    "bought": "purchase", "trip": "travel", "study": "learning", "chat": "conversation", "call": "meeting",
}
CAREER_KINDS = {"applied", "interview", "offer", "rejection"}
CAREER_TOPIC = "Job search"


def _kind(v) -> str:
    k = re.sub(r"[^a-z]+", "_", str(v or "other").lower()).strip("_")
    return k if k in EPISODE_KINDS else _KIND_SYNONYMS.get(k, "other")


_NO_OUTCOME = {"", "pending", "unknown", "none", "null", "n/a", "na", "tbd"}
_KIND_OUTCOME = {"rejection": "rejected", "offer": "offer"}


def _outcome(v, kind) -> str | None:
    """Small models stamp 'pending' on everything; keep only real results."""
    out = str(v or "").strip().lower()[:100]
    return _KIND_OUTCOME.get(_kind(kind)) or (None if out in _NO_OUTCOME else out)


def _imp(v) -> int:
    return v if isinstance(v, int) and 1 <= v <= 5 else 3


def _when(v, default: datetime) -> datetime:
    """LLM dates are in the user's local terms: naive → USER_TZ; date-only → that day at the reference local time."""
    if isinstance(v, str) and v.strip():
        try:
            d = datetime.fromisoformat(v.strip().replace("Z", "+00:00"))
            if len(v.strip()) == 10:
                return datetime.combine(d.date(), default.astimezone(USER_TZ).timetz())
            return d if d.tzinfo else d.replace(tzinfo=USER_TZ)
        except ValueError:
            pass
    return default


def _names(v) -> list[str]:
    return [n.strip() for n in v if isinstance(n, str) and n.strip()] if isinstance(v, list) else []


def _list(data: dict, key: str) -> list[dict]:
    v = data.get(key) if isinstance(data, dict) else None
    return [x for x in v if isinstance(x, dict)] if isinstance(v, list) else []


def extract_prompt(text: str, occurred_at: datetime, known: list[str] | None = None) -> str:
    return PROMPT.format(now=occurred_at.astimezone(USER_TZ).isoformat(), known=", ".join(known or []) or "none",
                         text=text)


async def extract(text: str, occurred_at: datetime, known: list[str] | None = None) -> Extraction:
    data = await complete_json(extract_prompt(text, occurred_at, known), system=SYSTEM, task="extract")
    return parse_extraction(data, occurred_at, text)


_APPLIED = re.compile(r"\b(appl(?:ied|ying)|applications? (?:was |were |has been |have been )?(?:submitted|sent)|"
                      r"(?:submitted|sent|filed) (?:my |the |an |his |her |their |a )?(?:applications?|resume|cv)|"
                      r"put in (?:an |my )?application)\b", re.I)
_RELATIVE = re.compile(r"\b(today|yesterday|tonight|this (?:morning|afternoon|evening|week)|last (?:week|night|month)|"
                       r"(?:mon|tues|wednes|thurs|fri|satur|sun)day|ago)\b", re.I)


def _segment(text: str, names: list[str]) -> str:
    """The paragraphs of `text` that mention any of `names` (the whole text if none do)."""
    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    hits = [p for p in paras if any(n and n.lower() in p.lower() for n in names)]
    return "\n\n".join(hits) or text


def _date_mentioned(d: datetime, segment: str) -> bool:
    day = d.astimezone(USER_TZ).date()
    forms = {day.isoformat(), f"{day.day} {day:%b}", f"{day:%b} {day.day}", f"{day.day} {day:%B}", f"{day:%B} {day.day}",
             f"{day:%d/%m/%Y}", f"{day.day}/{day.month}"}
    low = segment.lower()
    return any(f.lower() in low for f in forms) or bool(_RELATIVE.search(segment))


def _grounded(ep: dict, text: str, occurred_at: datetime) -> bool:
    """Models invent events and dates when handed a recap (seen: every company in a history summary became
    "Applied to X", each with a made-up date). Keep an episode only if the text backs it up."""
    seg = _segment(text, ep["entities"])
    if ep["kind"] == "applied" and not _APPLIED.search(seg):
        return False
    same_day = ep["occurred_at"].astimezone(USER_TZ).date() == occurred_at.astimezone(USER_TZ).date()
    return same_day or _date_mentioned(ep["occurred_at"], seg)


def parse_extraction(data, occurred_at: datetime, text: str | None = None) -> Extraction:
    """Model JSON -> a cleaned Extraction (shared by live calls and batch results). With `text`, episodes the
    text doesn't support (invented applications, invented dates) are dropped."""
    ex = Extraction()
    ex.entities = [{"name": e["name"].strip(), "kind": str(e.get("kind") or "topic").lower()}
                   for e in _list(data, "entities") if isinstance(e.get("name"), str) and e["name"].strip()]
    for e in _list(data, "episodes"):
        if not isinstance(e.get("summary"), str) or not e["summary"].strip():
            continue
        s = e.get("sentiment")
        ex.episodes.append({
            "kind": _kind(e.get("kind")),
            "summary": e["summary"].strip(),
            "occurred_at": _when(e.get("when"), occurred_at),
            "entities": _names(e.get("entities")),
            "outcome": _outcome(e.get("outcome"), e.get("kind")),
            "sentiment": float(s) if isinstance(s, (int, float)) else None,
            "importance": _imp(e.get("importance")),
        })
    # one career event per company, whatever the model did: counts ("how many applications") depend on it
    companies = {e["name"] for e in ex.entities if e["kind"] == "company"}
    split = []
    for e in ex.episodes:
        named = [n for n in e["entities"] if n in companies]
        if e["kind"] in CAREER_KINDS and len(named) > 1:
            rest = [n for n in e["entities"] if n not in companies]
            split += [{**e, "entities": [c, *rest]} for c in named]
        else:
            split.append(e)
    ex.episodes = split
    if text:
        ex.episodes = [e for e in ex.episodes if _grounded(e, text, occurred_at)]
    # career events always group under one topic, whatever the model remembered to tag
    for e in ex.episodes:
        if e["kind"] in CAREER_KINDS and CAREER_TOPIC not in e["entities"]:
            e["entities"].append(CAREER_TOPIC)
            if not any(x["name"] == CAREER_TOPIC for x in ex.entities):
                ex.entities.append({"name": CAREER_TOPIC, "kind": "topic"})
    ex.facts = [{"text": f["text"].strip(), "kind": f.get("kind") if f.get("kind") in ("fact", "preference") else "fact",
                 "entities": _names(f.get("entities")), "importance": _imp(f.get("importance"))}
                for f in _list(data, "facts") if isinstance(f.get("text"), str) and f["text"].strip()]
    ex.procedures = [{"name": str(p.get("name") or "procedure"), "steps": _names(p.get("steps")),
                      "entities": _names(p.get("entities"))}
                     for p in _list(data, "procedures") if len(_names(p.get("steps"))) >= 2]  # 1 step = a to-do
    ex.session_notes = [{"key": str(n["key"])[:255], "value": str(n["value"])}
                        for n in _list(data, "session_notes") if n.get("key") and n.get("value")]
    return ex
