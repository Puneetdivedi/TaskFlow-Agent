"""Long-term memory tools — let the agent remember and recall durable facts."""

from __future__ import annotations

import asyncio
from typing import Any

from src.interfaces import IFactStore, format_facts
from src.tools.base import Tool, ToolError


class RememberTool(Tool):
    """Store a durable fact in cross-session memory."""

    def __init__(self, store: IFactStore) -> None:
        self._store = store

    @property
    def name(self) -> str:
        return "remember"

    @property
    def description(self) -> str:
        return (
            "Store a durable fact in long-term memory. Facts persist across "
            "sessions and are shown to you in future turns, so use this for "
            "user preferences, project decisions, or anything worth "
            "remembering later. Retrieve facts with the 'recall' tool."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The fact to remember."},
                "topic": {
                    "type": "string",
                    "description": "Optional topic/category to group related facts.",
                },
            },
            "required": ["content"],
        }

    async def run(self, content: str, topic: str = "", **kwargs: Any) -> str:  # type: ignore[override]
        try:
            fact = await asyncio.to_thread(self._store.add, content, topic)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        return f"Remembered fact {fact.id}: {fact.content}"


class RecallTool(Tool):
    """Search long-term memory for facts matching a query."""

    def __init__(self, store: IFactStore) -> None:
        self._store = store

    @property
    def name(self) -> str:
        return "recall"

    @property
    def description(self) -> str:
        return (
            "Search long-term memory for facts matching a query string. Call "
            "with no query to list your most recent memories. Returns the "
            "matching facts with their ids, topics, and contents."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Text to match against fact content or topic "
                    "(omit to list recent memories).",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Maximum number of results (default 10).",
                },
            },
        }

    async def run(self, query: str = "", limit: int = 10, **kwargs: Any) -> str:
        try:
            facts = await asyncio.to_thread(self._store.search, query, limit)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        return format_facts(facts)
