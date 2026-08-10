"""Protocol for persistent cross-session fact storage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Fact:
    """A single durable memory persisted across sessions."""

    id: str
    content: str
    topic: str
    created_at: str  # ISO-8601 timestamp


class IFactStore(Protocol):
    """Interface for storing and retrieving durable facts.

    Facts are global — not tied to a named session — so they survive
    restarts and are shared across all conversations.
    """

    def add(self, content: str, topic: str = "") -> Fact:
        """Store a new fact and return it."""
        ...

    def search(self, query: str = "", limit: int = 10) -> list[Fact]:
        """Return facts whose content or topic matches *query*, newest first."""
        ...

    def list(self, limit: int = 50) -> list[Fact]:
        """Return the most recently added facts, newest first."""
        ...

    def get(self, fact_id: str) -> Fact:
        """Return the fact with *fact_id*.

        Raises ``KeyError`` if no such fact exists.
        """
        ...

    def delete(self, fact_id: str) -> None:
        """Remove the fact with *fact_id*.

        Raises ``KeyError`` if no such fact exists.
        """
        ...


def format_facts(facts: list[Fact]) -> str:
    """Render *facts* for the user or the LLM."""
    if not facts:
        return "No memories found."
    label = "memory" if len(facts) == 1 else "memories"
    lines = [f"{len(facts)} {label}:"]
    for fact in facts:
        prefix = f"{fact.topic}: " if fact.topic else ""
        lines.append(f"  [{fact.id}] {prefix}{fact.content}")
    return "\n".join(lines)
