"""Persistent session storage — saves conversation threads in SQLite.

Sessions live in the ``sessions`` table of the shared TaskFlow database
(default ``~/.taskflow/taskflow.db``), mirroring the ``TaskStore`` and
``FileIndex`` persistence pattern.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from src.interfaces.session_store import SessionInfo
from src.interfaces.usage import Usage
from src.memory.sqlite_store import (
    MEMORY_SUMMARIES_SCHEMA,
    SESSION_USAGE_SCHEMA,
    SESSIONS_SCHEMA,
    SQLiteStore,
)

logger = logging.getLogger(__name__)

# Names are slugs: start with an alphanumeric, then letters/digits/._-.
# This keeps names filesystem-safe and prevents path traversal.
_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_MAX_NAME_LEN = 64


class SessionStore(SQLiteStore):
    """SQLite-backed store for named conversation sessions.

    Each session also carries an optional rolling ``summary`` (the semantic
    memory of older turns) persisted in the ``memory_summaries`` table and
    restored on load.
    """

    _SCHEMA = SESSIONS_SCHEMA + MEMORY_SUMMARIES_SCHEMA + SESSION_USAGE_SCHEMA

    def __init__(self, db_path: Path | None = None) -> None:
        super().__init__(db_path)

    # ------------------------------------------------------------------
    def _validate_name(self, name: str) -> None:
        if not name:
            raise ValueError("Session name must not be empty")
        if len(name) > _MAX_NAME_LEN:
            raise ValueError(f"Session name too long (max {_MAX_NAME_LEN} chars): {name!r}")
        if not _NAME_RE.fullmatch(name):
            raise ValueError(
                "Session name must contain only letters, digits, '.', '_', '-' "
                f"and start with a letter or digit — got {name!r}"
            )

    # ------------------------------------------------------------------
    def save(
        self,
        name: str,
        messages: list[dict[str, Any]],
        summary: str = "",
        usage: Usage | None = None,
    ) -> None:
        """Persist *messages*, the rolling *summary*, and optional *usage*.

        All tables are written in the same transaction, so a session, its
        summary, and its cost are always consistent on disk. The usage row is
        only written when *usage* is provided, so sessions saved without
        billing data keep ``load_usage`` returning ``None``.
        """
        self._validate_name(name)
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO sessions (name, saved_at, message_count, messages) "
                "VALUES (?, ?, ?, ?)",
                (name, datetime.now().isoformat(), len(messages), json.dumps(messages)),
            )
            conn.execute(
                "INSERT OR REPLACE INTO memory_summaries (name, summary) VALUES (?, ?)",
                (name, summary or ""),
            )
            if usage is not None:
                conn.execute(
                    "INSERT OR REPLACE INTO session_usage "
                    "(name, input_tokens, output_tokens, cache_read_input_tokens, "
                    "cache_creation_input_tokens) VALUES (?, ?, ?, ?, ?)",
                    (
                        name,
                        usage.input_tokens,
                        usage.output_tokens,
                        usage.cache_read_input_tokens,
                        usage.cache_creation_input_tokens,
                    ),
                )
        logger.info("Saved session %r with %d message(s)", name, len(messages))

    def load(self, name: str) -> list[dict[str, Any]]:
        """Return the messages saved under *name*.

        Raises ``KeyError`` if the session does not exist and ``ValueError``
        if the stored record is corrupted.
        """
        self._validate_name(name)
        with self._connect() as conn:
            row = conn.execute("SELECT messages FROM sessions WHERE name = ?", (name,)).fetchone()
        if row is None:
            raise KeyError(f"Session {name!r} not found")
        try:
            messages = json.loads(row["messages"])
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f"Session {name!r} is corrupted: {exc}") from exc
        if not isinstance(messages, list):
            raise ValueError(f"Session {name!r} is corrupted: missing messages list")
        return messages

    def load_summary(self, name: str) -> str:
        """Return the rolling summary saved for *name* (``""`` if none)."""
        self._validate_name(name)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT summary FROM memory_summaries WHERE name = ?", (name,)
            ).fetchone()
        return row["summary"] if row is not None else ""

    def load_usage(self, name: str) -> Usage | None:
        """Return the token usage saved for *name* (``None`` if none)."""
        self._validate_name(name)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT input_tokens, output_tokens, cache_read_input_tokens, "
                "cache_creation_input_tokens FROM session_usage WHERE name = ?",
                (name,),
            ).fetchone()
        if row is None:
            return None
        return Usage(
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            cache_read_input_tokens=row["cache_read_input_tokens"],
            cache_creation_input_tokens=row["cache_creation_input_tokens"],
        )

    def delete(self, name: str) -> None:
        """Remove the saved session *name*, its summary, and its usage.

        Raises ``KeyError`` if the session does not exist.
        """
        self._validate_name(name)
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM sessions WHERE name = ?", (name,))
            conn.execute("DELETE FROM memory_summaries WHERE name = ?", (name,))
            conn.execute("DELETE FROM session_usage WHERE name = ?", (name,))
        if cursor.rowcount == 0:
            raise KeyError(f"Session {name!r} not found")
        logger.info("Deleted session %r", name)

    def exists(self, name: str) -> bool:
        """Return whether a session named *name* exists."""
        try:
            self._validate_name(name)
        except ValueError:
            return False
        with self._connect() as conn:
            row = conn.execute("SELECT 1 FROM sessions WHERE name = ?", (name,)).fetchone()
        return row is not None

    def list(self) -> list[SessionInfo]:
        """Return metadata for all saved sessions, newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT name, saved_at, message_count FROM sessions ORDER BY saved_at DESC"
            ).fetchall()
        return [
            SessionInfo(
                name=row["name"],
                message_count=row["message_count"],
                updated_at=row["saved_at"],
            )
            for row in rows
        ]

    def latest(self) -> str | None:
        """Return the name of the most recently saved session, or ``None``."""
        infos = self.list()
        return infos[0].name if infos else None
