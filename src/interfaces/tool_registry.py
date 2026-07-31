"""Protocol for the tool registry."""

from __future__ import annotations

from typing import Any, Protocol


class IToolRegistry(Protocol):
    """Interface for registering, listing, and dispatching tools."""

    @property
    def tool_names(self) -> list[str]:
        """Return the names of all registered tools."""
        ...

    def anthropic_tool_defs(self) -> list[dict[str, Any]]:
        """Return tool definitions in Anthropic's tool-use API format."""
        ...

    async def dispatch(self, name: str, arguments: dict[str, Any]) -> str:
        """Look up a tool by *name*, call it with *arguments*, return its text result."""
        ...
