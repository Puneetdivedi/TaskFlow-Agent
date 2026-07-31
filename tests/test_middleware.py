"""Tests for the middleware pipeline."""

from __future__ import annotations

from typing import Any

import pytest

from src.tools.base import ToolError
from src.tools.middleware import (
    AuditMiddleware,
    LoggingMiddleware,
    SafetyMiddleware,
    ToolMiddleware,
    ToolPipeline,
)


class _AppendCallMiddleware(ToolMiddleware):
    """Testing middleware that records calls and optionally modifies args."""

    def __init__(self, name: str, fail_before: bool = False) -> None:
        self.name = name
        self.fail_before = fail_before
        self.calls: list[str] = []

    async def before(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(f"before:{self.name}")
        if self.fail_before:
            raise ToolError(f"Blocked by {self.name}")
        return arguments

    async def after(self, tool_name: str, result: str, exception: BaseException | None) -> str:
        self.calls.append(f"after:{self.name}")
        return result


class TestToolPipeline:
    async def test_middleware_runs_in_order(self) -> None:
        a = _AppendCallMiddleware("A")
        b = _AppendCallMiddleware("B")
        pipeline = ToolPipeline(middleware=[a, b])

        async def dispatch_fn(name: str, args: dict) -> str:
            return "done"

        result = await pipeline.run(dispatch_fn, "test_tool", {"arg": 1})
        assert result == "done"
        # Before hooks in forward order, after hooks in reverse
        assert a.calls == ["before:A", "after:A"]
        assert b.calls == ["before:B", "after:B"]

    async def test_middleware_aborts_on_before_failure(self) -> None:
        blocker = _AppendCallMiddleware("blocker", fail_before=True)
        pipeline = ToolPipeline(middleware=[blocker])

        async def dispatch_fn(name: str, args: dict) -> str:
            return "done"

        with pytest.raises(ToolError, match="Blocked by blocker"):
            await pipeline.run(dispatch_fn, "test_tool", {"arg": 1})

    async def test_middleware_after_still_runs_on_error(self) -> None:
        """after hooks should run even if dispatch raises."""
        mw = _AppendCallMiddleware("mw")
        pipeline = ToolPipeline(middleware=[mw])

        async def dispatch_fn(name: str, args: dict) -> str:
            raise RuntimeError("crash")

        with pytest.raises(RuntimeError, match="crash"):
            await pipeline.run(dispatch_fn, "test_tool", {"arg": 1})

        # after should still have been called (with the exception)
        assert "after:mw" in mw.calls

    async def test_empty_middleware_passthrough(self) -> None:
        pipeline = ToolPipeline()

        async def dispatch_fn(name: str, args: dict) -> str:
            return "pass"

        result = await pipeline.run(dispatch_fn, "t", {})
        assert result == "pass"


class TestSafetyMiddleware:
    async def test_blocks_disallowed_tool(self) -> None:
        mw = SafetyMiddleware(allowed_tools=["read_file"])
        pipeline = ToolPipeline(middleware=[mw])

        async def dispatch_fn(name: str, args: dict) -> str:
            return "ok"

        with pytest.raises(ToolError, match="SafetyMiddleware blocked"):
            await pipeline.run(dispatch_fn, "write_file", {})

    async def test_allows_allowed_tool(self) -> None:
        mw = SafetyMiddleware(allowed_tools=["read_file"])
        pipeline = ToolPipeline(middleware=[mw])

        async def dispatch_fn(name: str, args: dict) -> str:
            return "ok"

        result = await pipeline.run(dispatch_fn, "read_file", {})
        assert result == "ok"

    async def test_allows_all_when_unrestricted(self) -> None:
        mw = SafetyMiddleware()  # no allowed list = allow all
        pipeline = ToolPipeline(middleware=[mw])

        async def dispatch_fn(name: str, args: dict) -> str:
            return "ok"

        result = await pipeline.run(dispatch_fn, "anything", {})
        assert result == "ok"


class TestBuiltinMiddleware:
    async def test_logging_middleware_does_not_modify(self) -> None:
        mw = LoggingMiddleware()
        args = await mw.before("test", {"key": "val"})
        assert args == {"key": "val"}
        result = await mw.after("test", "output", None)
        assert result == "output"

    async def test_audit_middleware_passthrough(self) -> None:
        mw = AuditMiddleware()
        args = await mw.before("read_file", {"path": "x"})
        assert args == {"path": "x"}
        result = await mw.after("read_file", "content", None)
        assert result == "content"
