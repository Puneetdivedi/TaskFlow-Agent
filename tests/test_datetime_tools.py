"""Tests for the date & time tools."""

from __future__ import annotations

from datetime import datetime

import pytest

from src.tools.base import ToolError
from src.tools.datetime_tools import (
    CurrentDateTool,
    DateAddTool,
    DaysBetweenTool,
)


class TestCurrentDateTool:
    def test_name(self) -> None:
        assert CurrentDateTool().name == "current_date"

    async def test_local_returns_iso(self) -> None:
        result = await CurrentDateTool().run()
        dt = datetime.fromisoformat(result)
        assert dt.microsecond == 0

    async def test_utc_has_offset(self) -> None:
        result = await CurrentDateTool().run(tz="UTC")
        assert result.endswith("+00:00")

    async def test_named_timezone(self) -> None:
        result = await CurrentDateTool().run(tz="America/New_York")
        assert datetime.fromisoformat(result).tzinfo is not None

    async def test_invalid_timezone_raises(self) -> None:
        with pytest.raises(ToolError, match="Unknown timezone"):
            await CurrentDateTool().run(tz="Mars/Olympus_Mons")


class TestDateAddTool:
    def test_name(self) -> None:
        assert DateAddTool().name == "date_add"

    async def test_add_days(self) -> None:
        assert await DateAddTool().run("2026-08-10", days=5) == "2026-08-15"

    async def test_add_weeks(self) -> None:
        assert await DateAddTool().run("2026-08-10", weeks=1) == "2026-08-17"

    async def test_add_months_clamps_day(self) -> None:
        assert await DateAddTool().run("2026-01-31", months=1) == "2026-02-28"

    async def test_add_years_clamps_leap_day(self) -> None:
        assert await DateAddTool().run("2024-02-29", years=1) == "2025-02-28"

    async def test_subtract_months(self) -> None:
        assert await DateAddTool().run("2026-03-15", months=-1) == "2026-02-15"

    async def test_keeps_time_component(self) -> None:
        assert await DateAddTool().run("2026-08-10T10:30:00", days=1) == "2026-08-11T10:30:00"

    async def test_invalid_date_raises(self) -> None:
        with pytest.raises(ToolError, match="Invalid ISO datetime"):
            await DateAddTool().run("not-a-date", days=1)

    async def test_non_integer_delta_raises(self) -> None:
        with pytest.raises(ToolError, match="days must be an integer"):
            await DateAddTool().run("2026-08-10", days="two")


class TestDaysBetweenTool:
    def test_name(self) -> None:
        assert DaysBetweenTool().name == "days_between"

    async def test_positive_days(self) -> None:
        assert await DaysBetweenTool().run("2026-08-10", "2026-08-15") == "5"

    async def test_negative_days(self) -> None:
        assert await DaysBetweenTool().run("2026-08-15", "2026-08-10") == "-5"

    async def test_same_day(self) -> None:
        assert await DaysBetweenTool().run("2026-08-10", "2026-08-10") == "0"

    async def test_ignores_time_of_day(self) -> None:
        assert await DaysBetweenTool().run("2026-08-10T23:59:00", "2026-08-10T00:01:00") == "0"

    async def test_invalid_date_raises(self) -> None:
        with pytest.raises(ToolError, match="Invalid ISO datetime"):
            await DaysBetweenTool().run("2026-13-01", "2026-08-10")
