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
from pathlib import Path
from typing import Any

from src.tools.base import ToolError
from src.tools.security import validate_path_safe
from src.tools.shell_tools import FORBIDDEN_PREFIXES

logger = logging.getLogger(__name__)

#: Default cap on the length of a single tool result, in characters. Longer
#: results are truncated so a huge tool output can't blow the model context.
DEFAULT_MAX_RESULT_CHARS = 20_000

#: Which tool arguments name file-system paths that must stay inside the
#: guardrail's allowed base directory. ``run_shell``'s ``work_dir`` is included
#: so a shell command can't be pointed at a directory outside the base.
PATH_ARG_NAMES: dict[str, tuple[str, ...]] = {
    "read_file": ("path",),
    "write_file": ("path",),
    "copy_file": ("source", "dest"),
    "delete_file": ("path",),
    "file_info": ("path",),
    "mkdir": ("path",),
    "move_file": ("source", "dest"),
    "list_files": ("path",),
    "search_files": ("path",),
    "yaml_read": ("path",),
    "yaml_write": ("path",),
    "json_read": ("path",),
    "json_write": ("path",),
    "csv_read": ("path",),
    "csv_aggregate": ("path",),
    "file_index": ("path",),
    "run_shell": ("work_dir",),
}


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

    AUDIT_TOOLS = {"delete_file", "move_file", "run_shell", "send_email"}

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
                f"SafetyMiddleware blocked: {tool_name!r} is not in the allowed tool list"
            )
        return arguments

    async def after(
        self,
        tool_name: str,
        result: str,
        exception: BaseException | None,
    ) -> str:
        return result


class GuardrailMiddleware(ToolMiddleware):
    """A configurable safety-policy layer gating every tool dispatch.

    Before dispatch it blocks dangerous shell commands and any path argument
    that escapes the allowed base directory; after dispatch it caps oversized
    results. Because it lives in the shared :class:`ToolPipeline`, it applies
    to the main agent *and* to sub-agents, which dispatch through the same
    registry.
    """

    def __init__(
        self,
        work_dir: Path | None = None,
        *,
        enabled: bool = True,
        max_result_chars: int = DEFAULT_MAX_RESULT_CHARS,
        deny_prefixes: tuple[str, ...] = tuple(FORBIDDEN_PREFIXES),
    ) -> None:
        self._work_dir = Path(work_dir).resolve() if work_dir else Path.cwd()
        self._enabled = enabled
        self._max_result_chars = max_result_chars
        self._deny_prefixes = deny_prefixes

    # ------------------------------------------------------------------
    async def before(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        if not self._enabled:
            return arguments

        if tool_name == "run_shell":
            command = arguments.get("command")
            if isinstance(command, str):
                self._check_command(command)

        for arg_name in PATH_ARG_NAMES.get(tool_name, ()):
            value = arguments.get(arg_name)
            if value is None:
                continue
            if isinstance(value, (str, Path)):
                validate_path_safe(Path(value), self._work_dir)

        return arguments

    def _check_command(self, command: str) -> None:
        """Raise ``ToolError`` if *command* starts with a denied prefix."""
        stripped = command.strip().lower()
        for forbidden in self._deny_prefixes:
            if stripped.startswith(forbidden):
                raise ToolError(f"Guardrail blocked command: {forbidden!r} is not allowed.")

    async def after(
        self,
        tool_name: str,
        result: str,
        exception: BaseException | None,
    ) -> str:
        if not self._enabled or self._max_result_chars <= 0 or exception is not None:
            return result
        if len(result) > self._max_result_chars:
            return (
                result[: self._max_result_chars]
                + f"\n…[truncated: {len(result)} chars, max {self._max_result_chars}]"
            )
        return result
