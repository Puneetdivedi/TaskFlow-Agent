"""Persistent cross-session fact storage in the shared SQLite database."""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from src.interfaces.fact_store import Fact
from src.memory.sqlite_store import FACTS_SCHEMA, SQLiteStore

logger = logging.getLogger(__name__)

_ID_RE = re.compile(r"^f(\d+)$")


class FactStore(SQLiteStore):
    """SQLite-backed store for durable cross-session facts."""

    _SCHEMA = FACTS_SCHEMA

    def __init__(self, db_path: Path | None = None) -> None:
        super().__init__(db_path)

    # ------------------------------------------------------------------
    def _next_id(self) -> str:
        with self._connect() as conn:
            rows = conn.execute("SELECT id FROM facts").fetchall()
        max_n = 0
        for row in rows:
            match = _ID_RE.match(row["id"])
            if match:
                max_n = max(max_n, int(match.group(1)))
        return f"f{max_n + 1}"

    @staticmethod
    def _now() -> str:
        return datetime.now().isoformat()

    @staticmethod
    def _validate_content(content: str) -> None:
        if not content.strip():
            raise ValueError("Memory content must not be empty")

    @staticmethod
    def _validate_limit(limit: int) -> None:
        """Reject non-ints and non-positive values (bools are a subclass of int)."""
        if type(limit) is not int or limit < 1:
            raise ValueError(f"Invalid limit {limit!r} — must be a positive integer")

    # ------------------------------------------------------------------
    def add(self, content: str, topic: str = "") -> Fact:
        self._validate_content(content)
        fact = Fact(
            id=self._next_id(),
            content=content.strip(),
            topic=topic.strip(),
            created_at=self._now(),
        )
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO facts (id, content, topic, created_at) VALUES (?, ?, ?, ?)",
                (fact.id, fact.content, fact.topic, fact.created_at),
            )
        logger.info("Remembered fact %s", fact.id)
        return fact

    def _query(self, where: str, params: tuple[Any, ...], limit: int) -> list[Fact]:
        self._validate_limit(limit)
        sql = "SELECT id, content, topic, created_at FROM facts"
        if where:
            sql += f" {where}"
        sql += " ORDER BY rowid DESC LIMIT ?"
        with self._connect() as conn:
            rows = conn.execute(sql, (*params, limit)).fetchall()
        return [
            Fact(
                id=row["id"],
                content=row["content"],
                topic=row["topic"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def search(self, query: str = "", limit: int = 10) -> list[Fact]:
        """Return facts whose content or topic matches *query* (newest first).

        Matching is a case-insensitive substring test over content and topic;
        an empty query lists the most recent facts.
        """
        if not query.strip():
            return self.list(limit)
        pattern = f"%{query.strip()}%"
        return self._query("WHERE content LIKE ? OR topic LIKE ?", (pattern, pattern), limit)

    def list(self, limit: int = 50) -> list[Fact]:
        """Return the most recently added facts, newest first."""
        return self._query("", (), limit)

    def get(self, fact_id: str) -> Fact:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, content, topic, created_at FROM facts WHERE id = ?",
                (fact_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Memory {fact_id!r} not found")
        return Fact(
            id=row["id"],
            content=row["content"],
            topic=row["topic"],
            created_at=row["created_at"],
        )

    def delete(self, fact_id: str) -> None:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
        if cursor.rowcount == 0:
            raise KeyError(f"Memory {fact_id!r} not found")
        logger.info("Forgot fact %s", fact_id)
