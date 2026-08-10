"""Approval interface for gating tool calls before they dispatch.

A tool approver is an async callback invoked immediately before each tool
call. It returns ``None`` to allow the call or a ``str`` denial reason to
block it — a blocked call is reported back to the model as a plain
``tool_result`` so the agent can adapt.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, TypeAlias

ToolApprover: TypeAlias = Callable[[str, dict[str, Any]], Awaitable[str | None]]
