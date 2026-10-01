"""Agentic memory: shared, typed, situational memory for every assistant."""
import os
from datetime import datetime
from zoneinfo import ZoneInfo

USER_TZ = ZoneInfo(os.environ.get("USER_TZ", "Asia/Kolkata"))


def local_date(dt: datetime) -> str:
    """Dates are shown in the user's timezone — 'today' must mean their today."""
    return dt.astimezone(USER_TZ).date().isoformat()
