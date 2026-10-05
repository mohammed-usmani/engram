"""Tracked dates: `Key: value` fields in a document (Expires:, Renews:, Due:, Birthday:...)
surface as 'Coming up' in recall, the upcoming() tool and the /memory page.

Computed from doc_metadata on every call — no table to keep in sync.
Privacy: only a reminder line (title, label, date) ever leaves the document, never its content;
local-only documents count only for local requests.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models import ContextDocument
from src.privacy import readable

ONE_OFF = {
    **dict.fromkeys(("expires", "expiry", "expiry date", "expiration", "expiration date", "valid until"), "expires"),
    **dict.fromkeys(("renews", "renewal", "renewal date"), "renews"),
    **dict.fromkeys(("due", "due date", "deadline"), "due"),
    "appointment": "appointment", "next service": "next service", "matures": "matures",
}
YEARLY = {**dict.fromkeys(("birthday", "date of birth", "dob"), "birthday"), "anniversary": "anniversary"}
DATE_RULE = ("Tracked dates come from `Key: value` lines at the top of a document. One-off: "
             + ", ".join(sorted({k.title() for k in ONE_OFF})) + ". Yearly: "
             + ", ".join(sorted({k.title() for k in YEARLY})) + ". Values like 2027-03-14, 14 Mar 2027, "
             "March 14, 2027 or 14/03/2027 (day first); a yearly date may omit the year ('10 Oct').")

_FORMATS = ("%Y-%m-%d", "%d %b %Y", "%d %B %Y", "%b %d %Y", "%B %d %Y", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y")
_YEARLESS = ("%d %b", "%d %B", "%b %d", "%B %d")
_OVERDUE_DAYS = 30


def parse_date(value: str) -> tuple[date | None, bool]:
    """(date, yearless). A yearless date gets year 2000 (a leap year, so 29 Feb survives)."""
    s = re.split(r"[(;]", value or "")[0]
    s = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", s).replace(",", " ")
    s = " ".join(s.split())
    for fmt in _FORMATS:
        try:
            return datetime.strptime(s, fmt).date(), False
        except ValueError:
            pass
    for fmt in _YEARLESS:
        try:
            return datetime.strptime(f"{s} 2000", f"{fmt} %Y").date(), True
        except ValueError:
            pass
    return None, False


def _today() -> date:
    return date.today()


def _next_yearly(d: date, today: date) -> date:
    for year in (today.year, today.year + 1):
        try:
            nxt = d.replace(year=year)
        except ValueError:  # 29 Feb in a non-leap year
            nxt = date(year, 2, 28)
        if nxt >= today:
            return nxt
    raise AssertionError("unreachable")


async def upcoming(session: AsyncSession, days: int = 60, today: date | None = None) -> tuple[list[dict], list[dict]]:
    """(items due within `days` plus one-offs overdue by up to 30 days, sorted by date;
    date fields that could not be read)."""
    today = today or _today()
    end = today + timedelta(days=days)
    docs = (await session.execute(select(ContextDocument).where(readable(ContextDocument.privacy)))).scalars()
    items, unreadable = [], []
    for doc in docs:
        for key, value in (doc.doc_metadata or {}).items():
            k = key.strip().lower()
            if k not in ONE_OFF and k not in YEARLY:
                continue
            d, yearless = parse_date(value)
            if d is None:
                unreadable.append({"slug": doc.slug, "field": key, "value": value})
                continue
            item = {"slug": doc.slug, "title": doc.title, "privacy": doc.privacy or "normal"}
            if k in YEARLY:
                when = _next_yearly(d, today)
                item["label"] = YEARLY[k]
                if not yearless:
                    item["turns"] = when.year - d.year
            else:
                when, item["label"] = d, ONE_OFF[k]
                if when < today - timedelta(days=_OVERDUE_DAYS):
                    continue
            if when > end:
                continue
            items.append({**item, "date": when.isoformat(), "days_left": (when - today).days})
    items.sort(key=lambda i: i["date"])
    return items, unreadable


def format_line(item: dict) -> str:
    n = item["days_left"]
    when = "today" if n == 0 else "tomorrow" if n == 1 else f"in {n} days" if n > 0 else f"OVERDUE by {-n} days"
    turns = f", turns {item['turns']}" if "turns" in item else ""
    return f"- {item['title']} — {item['label']} {item['date']} ({when}{turns}) [d:{item['slug']}]"
