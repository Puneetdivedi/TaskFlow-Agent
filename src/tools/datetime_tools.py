"""Date & time tools — current time in any timezone, date arithmetic, day spans."""

from __future__ import annotations

import calendar
import logging
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from src.tools.base import Tool, ToolError

logger = logging.getLogger(__name__)

#: Constants exposed to ``date_add``-adjacent callers via the schema defaults.
_LOCAL_TZ = "local"


def _parse_datetime(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ToolError(f"Invalid ISO datetime: {value!r} ({exc})") from exc


def _add_months(dt: datetime, months: int) -> datetime:
    """Add *months* to *dt*, clamping the day to the target month's length."""
    total = dt.year * 12 + (dt.month - 1) + months
    year, month0 = divmod(total, 12)
    month = month0 + 1
    day = min(dt.day, calendar.monthrange(year, month)[1])
    return dt.replace(year=year, month=month, day=day)


def _require_int(kwargs: dict[str, Any], key: str, default: int) -> int:
    value = kwargs.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ToolError(f"{key} must be an integer")
    return value


class CurrentDateTool(Tool):
    """Return the current date/time, optionally in a named timezone."""

    @property
    def name(self) -> str:
        return "current_date"

    @property
    def description(self) -> str:
        return (
            "Return the current date and time as ISO-8601. "
            "tz is a timezone name (e.g. America/New_York, UTC) or 'local' (default)."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "tz": {"type": "string", "description": "IANA timezone name or 'local'"},
            },
        }

    async def run(self, **kwargs: Any) -> str:
        tz_name = kwargs.get("tz", _LOCAL_TZ)
        if not isinstance(tz_name, str):
            raise ToolError("tz must be a string")
        if tz_name == _LOCAL_TZ:
            now = datetime.now()
        else:
            try:
                now = datetime.now(ZoneInfo(tz_name))
            except (ZoneInfoNotFoundError, KeyError) as exc:
                raise ToolError(f"Unknown timezone: {tz_name!r}") from exc
        return now.replace(microsecond=0).isoformat(timespec="seconds")


class DateAddTool(Tool):
    """Add days/weeks/months/years to a date."""

    @property
    def name(self) -> str:
        return "date_add"

    @property
    def description(self) -> str:
        return (
            "Add days, weeks, months, and/or years to an ISO-8601 date. "
            "Month arithmetic clamps to the target month's length (e.g. Jan 31 + 1 month = Feb 28)."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "ISO-8601 date (e.g. 2026-08-10)"},
                "days": {"type": "integer", "description": "Days to add (default: 0)"},
                "weeks": {"type": "integer", "description": "Weeks to add (default: 0)"},
                "months": {"type": "integer", "description": "Months to add (default: 0)"},
                "years": {"type": "integer", "description": "Years to add (default: 0)"},
            },
            "required": ["date"],
        }

    async def run(self, date: str, **kwargs: Any) -> str:  # type: ignore[override]
        dt = _parse_datetime(date)
        has_time = "T" in date or " " in date
        years = _require_int(kwargs, "years", 0)
        months = _require_int(kwargs, "months", 0)
        days = _require_int(kwargs, "days", 0)
        weeks = _require_int(kwargs, "weeks", 0)
        if years:
            dt = _add_months(dt, years * 12)
        if months:
            dt = _add_months(dt, months)
        if days or weeks:
            dt = dt + timedelta(days=days + weeks * 7)
        dt = dt.replace(microsecond=0)
        if has_time:
            return dt.isoformat(timespec="seconds")
        return dt.date().isoformat()


class DaysBetweenTool(Tool):
    """Return the number of days between two ISO dates."""

    @property
    def name(self) -> str:
        return "days_between"

    @property
    def description(self) -> str:
        return (
            "Return the number of days between two ISO-8601 dates "
            "(positive when end is after start, negative otherwise)."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "start": {"type": "string", "description": "ISO-8601 start date"},
                "end": {"type": "string", "description": "ISO-8601 end date"},
            },
            "required": ["start", "end"],
        }

    async def run(self, start: str, end: str, **kwargs: Any) -> str:  # type: ignore[override]
        start_date: date = _parse_datetime(start).date()
        end_date: date = _parse_datetime(end).date()
        return str((end_date - start_date).days)
