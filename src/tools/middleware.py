"""Middleware pipeline for tool dispatch — cross-cutting concerns.

Middleware hooks run **before** and **after** every tool dispatch,
allowing logging, auditing, safety checks, and other concerns to be
composed without touching the tool implementations themselves.

Usage::

    pipeline = ToolPipeline(middleware=[LoggingMiddleware(), AuditMiddleware()])
    result = await pipeline.run("read_file", {"path": "..."})
"""

from __future__ import annotations

import abc
import logging
from typing import Any

from src.tools.base import ToolError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Middleware base
# ---------------------------------------------------------------------------
class ToolMiddleware(abc.ABC):
    """Abstract middleware — override ``before`` and / or ``after``."""

    @abc.abstractmethod
    async def before(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Called **before** the tool runs.

        Returns the (possibly modified) arguments dict.
        Raise ``ToolError`` to abort execution.
        """
        return arguments

    @abc.abstractmethod
    async def after(
        self,
        tool_name: str,
        result: str,
        exception: BaseException | None,
    ) -> str:
        """Called **after** the tool runs.

        *exception* is ``None`` on success.  Return the (possibly
        modified) result string; a replacement is returned to the caller.
        """
        return result


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class ToolPipeline:
    """Runs a chain of middleware around a tool-dispatch callable."""

    def __init__(self, middleware: list[ToolMiddleware] | None = None) -> None:
        self._middleware = middleware or []

    async def run(self, dispatch_fn: Any, tool_name: str, arguments: dict[str, Any]) -> str:
        """Run *dispatch_fn* wrapped in the middleware chain."""
        return await self._dispatch(dispatch_fn, tool_name, arguments)

    def wrap(self, dispatch_fn: Any) -> Any:
        """Return an ``async (name, args) -> str`` wrapper."""

        async def _dispatched(name: str, args: dict[str, Any]) -> str:
            return await self._dispatch(dispatch_fn, name, args)

        return _dispatched

    async def _dispatch(
        self,
        dispatch_fn: Any,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:
        # --- before chain (forward order) ---
        modified_args = arguments
        for mw in self._middleware:
            modified_args = await mw.before(tool_name, modified_args)

        # --- actual tool dispatch ---
        result: str = ""
        exc: BaseException | None = None
        try:
            result = await dispatch_fn(tool_name, modified_args)
        except ToolError:
            raise
        except Exception as e:
            exc = e
            raise
        finally:
            # --- after chain (reverse order) ---
            for mw in reversed(self._middleware):
                result = await mw.after(tool_name, result, exc)

        return result


# ---------------------------------------------------------------------------
# Built-in middleware implementations
# ---------------------------------------------------------------------------
class LoggingMiddleware(ToolMiddleware):
    """Log every tool dispatch at DEBUG level."""

    async def before(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        logger.debug("Tool dispatch: %s args=%s", tool_name, arguments)
        return arguments

    async def after(
        self,
        tool_name: str,
        result: str,
        exception: BaseException | None,
    ) -> str:
        if exception:
            logger.warning("Tool %s failed: %s", tool_name, exception)
        else:
            logger.debug("Tool %s returned %d chars", tool_name, len(result))
        return result


class AuditMiddleware(ToolMiddleware):
    """Record high-severity operations (delete, move, shell) at WARNING level."""

    AUDIT_TOOLS = {"delete_file", "move_file", "run_shell"}

    async def before(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        if tool_name in self.AUDIT_TOOLS:
            logger.warning(
                "AUDIT: %s invoked with args: %s",
                tool_name,
                arguments,
            )
        return arguments

    async def after(
        self,
        tool_name: str,
        result: str,
        exception: BaseException | None,
    ) -> str:
        return result


class SafetyMiddleware(ToolMiddleware):
    """Abort dispatch if the tool name is not in the allowed list."""

    def __init__(self, allowed_tools: list[str] | None = None) -> None:
        self._allowed = allowed_tools

    async def before(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        if self._allowed is not None and tool_name not in self._allowed:
            raise ToolError(
                f"SafetyMiddleware blocked: {tool_name!r} is not in "
                f"the allowed tool list"
            )
        return arguments

    async def after(
        self,
        tool_name: str,
        result: str,
        exception: BaseException | None,
    ) -> str:
        return result
