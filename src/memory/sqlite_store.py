"""Shared SQLite persistence backend for the TaskFlow stores.

All three persistent stores (``TaskStore``, ``SessionStore``, ``FileIndex``)
back onto a single SQLite database file (``~/.taskflow/taskflow.db`` by
default).  Connections are opened fresh per operation so the stores are safe
to call from worker threads (``asyncio.to_thread``) without sharing
connection state, and a corrupt database file is recovered by recreating it
empty rather than crashing.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path.home() / ".taskflow" / "taskflow.db"

# --- Table schemas (one table per store) ----------------------------------
TASKS_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'todo',
    priority TEXT NOT NULL DEFAULT 'medium',
    created_at TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT '',
    due_at TEXT NOT NULL DEFAULT '',
    every_days INTEGER NOT NULL DEFAULT 0
);
"""

SESSIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    name TEXT PRIMARY KEY,
    saved_at TEXT NOT NULL,
    message_count INTEGER NOT NULL DEFAULT 0,
    messages TEXT NOT NULL
);
"""

INDEX_SCHEMA = """
CREATE TABLE IF NOT EXISTS index_roots (
    root TEXT PRIMARY KEY,
    file_count INTEGER NOT NULL DEFAULT 0,
    dir_count INTEGER NOT NULL DEFAULT 0,
    total_size INTEGER NOT NULL DEFAULT 0,
    indexed_at TEXT NOT NULL DEFAULT '',
    entries TEXT NOT NULL DEFAULT '{}'
);
"""

MEMORY_SUMMARIES_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_summaries (
    name TEXT PRIMARY KEY,
    summary TEXT NOT NULL DEFAULT ''
);
"""

SESSION_USAGE_SCHEMA = """
CREATE TABLE IF NOT EXISTS session_usage (
    name TEXT PRIMARY KEY,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cache_read_input_tokens INTEGER NOT NULL DEFAULT 0,
    cache_creation_input_tokens INTEGER NOT NULL DEFAULT 0
);
"""

ALL_SCHEMA = (
    TASKS_SCHEMA + SESSIONS_SCHEMA + INDEX_SCHEMA + MEMORY_SUMMARIES_SCHEMA + SESSION_USAGE_SCHEMA
)


class SQLiteStore:
    """Base class for SQLite-backed stores.

    Subclasses set ``_SCHEMA`` (the DDL that creates their table) and use
    :meth:`_connect` for every database operation.
    """

    _SCHEMA = ""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Yield a fresh connection that is committed on success, closed always."""
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        """Create this store's table, recovering from a corrupt database file."""
        try:
            with self._connect() as conn:
                conn.executescript(self._SCHEMA)
        except sqlite3.DatabaseError as exc:
            logger.warning("Database %s is corrupt — recreating it: %s", self._db_path, exc)
            self._db_path.unlink(missing_ok=True)
            with self._connect() as conn:
                conn.executescript(self._SCHEMA)
