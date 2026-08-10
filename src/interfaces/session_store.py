"""Protocol for persistent session storage (save/load conversation threads)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from src.interfaces.usage import Usage


@dataclass(frozen=True)
class SessionInfo:
    """Metadata about a saved session (no message bodies)."""

    name: str
    message_count: int
    updated_at: str  # ISO-8601 timestamp of the last save


class ISessionStore(Protocol):
    """Interface for persisting named conversation sessions.

    Sessions are identified by a name and stored on disk so history can
    survive process restarts.
    """

    def save(
        self,
        name: str,
        messages: list[dict[str, Any]],
        summary: str = "",
        usage: Usage | None = None,
    ) -> None:
        """Persist *messages*, the rolling *summary*, and optional *usage*."""
        ...

    def load(self, name: str) -> list[dict[str, Any]]:
        """Return the messages saved under *name*.

        Raises ``KeyError`` if no such session exists.
        """
        ...

    def load_summary(self, name: str) -> str:
        """Return the rolling summary saved for *name* (``""`` if none)."""
        ...

    def load_usage(self, name: str) -> Usage | None:
        """Return the token usage saved for *name* (``None`` if none)."""
        ...

    def delete(self, name: str) -> None:
        """Remove the saved session *name*.

        Raises ``KeyError`` if no such session exists.
        """
        ...

    def list(self) -> list[SessionInfo]:
        """Return metadata for all saved sessions, newest first."""
        ...

    def exists(self, name: str) -> bool:
        """Return whether a session named *name* exists."""
        ...

    def latest(self) -> str | None:
        """Return the name of the most recently saved session, or ``None``."""
        ...
