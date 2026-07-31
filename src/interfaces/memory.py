"""Protocol for conversation memory backends."""

from __future__ import annotations

from typing import Any, Protocol


class IMemory(Protocol):
    """Interface for storing and retrieving conversation history."""

    @property
    def messages(self) -> list[dict[str, Any]]:
        """Return all messages in Anthropic message format."""
        ...

    def add_user(self, content: str) -> None:
        """Append a user message."""
        ...

    def add_assistant(self, content: str | list[dict]) -> None:
        """Append an assistant message (text or content blocks)."""
        ...

    def add_tool_result(self, tool_use_id: str, content: str) -> None:
        """Append a tool-result payload as a user-style message."""
        ...

    def prune(self) -> None:
        """Drop oldest exchanges to stay within a token budget."""
        ...

    def clear(self) -> None:
        """Remove all messages."""
        ...
