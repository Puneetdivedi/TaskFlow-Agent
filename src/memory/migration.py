"""One-time migration of legacy JSON persistence files into SQLite.

The stores previously wrote JSON files (``~/.taskflow/tasks.json``, a
``sessions/*.json`` directory, and ``~/.taskflow/file_index.json``).  This
module imports any such files into the shared SQLite database on first run.
It is best-effort and idempotent: each table is only populated when it is
currently empty, and corrupted or malformed entries are skipped and logged —
it never raises.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path

from src.interfaces.task_store import Task
from src.memory.sqlite_store import ALL_SCHEMA

logger = logging.getLogger(__name__)

# The historical JSON locations replaced by the shared database.
_LEGACY_TASKS = Path.home() / ".taskflow" / "tasks.json"
_LEGACY_SESSIONS_DIR = Path.home() / ".taskflow" / "sessions"
_LEGACY_INDEX = Path.home() / ".taskflow" / "file_index.json"

_TASK_COLUMNS = (
    "id",
    "title",
    "description",
    "status",
    "priority",
    "created_at",
    "updated_at",
    "due_at",
    "every_days",
)


def migrate_legacy_data(
    db_path: Path | str,
    *,
    legacy_tasks: Path | str | None = None,
    legacy_sessions_dir: Path | str | None = None,
    legacy_index: Path | str | None = None,
) -> None:
    """Best-effort, idempotent import of legacy JSON data into *db_path*."""
    db = Path(db_path)
    tasks_src = Path(legacy_tasks) if legacy_tasks is not None else _LEGACY_TASKS
    sessions_src = (
        Path(legacy_sessions_dir) if legacy_sessions_dir is not None else _LEGACY_SESSIONS_DIR
    )
    index_src = Path(legacy_index) if legacy_index is not None else _LEGACY_INDEX

    if not any(src.exists() for src in (tasks_src, sessions_src, index_src)):
        return

    db.parent.mkdir(parents=True, exist_ok=True)
    try:
        conn = sqlite3.connect(db)
    except sqlite3.DatabaseError as exc:
        logger.warning("Cannot open database %s for legacy migration: %s", db, exc)
        return
    try:
        conn.executescript(ALL_SCHEMA)
        _import_tasks(conn, tasks_src)
        _import_sessions(conn, sessions_src)
        _import_index(conn, index_src)
        conn.commit()
    except sqlite3.DatabaseError as exc:
        logger.warning("Legacy migration failed for %s: %s", db, exc)
        conn.rollback()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Tracks
# ---------------------------------------------------------------------------
def _import_tasks(conn: sqlite3.Connection, path: Path) -> None:
    if not path.exists() or _count(conn, "tasks") > 0:
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Skipping legacy task file %s: %s", path, exc)
        return

    raw_tasks = payload.get("tasks", []) if isinstance(payload, dict) else []
    imported = 0
    for entry in raw_tasks:
        if not isinstance(entry, dict):
            continue
        try:
            task = Task(**entry)
        except (TypeError, ValueError):
            logger.warning("Skipping malformed legacy task entry: %s", entry)
            continue
        conn.execute(
            f"INSERT OR IGNORE INTO tasks ({', '.join(_TASK_COLUMNS)}) "
            f"VALUES ({', '.join('?' * len(_TASK_COLUMNS))})",
            tuple(task.__dict__[col] for col in _TASK_COLUMNS),
        )
        imported += 1
    if imported:
        logger.info("Migrated %d task(s) from %s", imported, path)


def _import_sessions(conn: sqlite3.Connection, path: Path) -> None:
    if not path.is_dir() or _count(conn, "sessions") > 0:
        return
    imported = 0
    for session_file in path.glob("*.json"):
        try:
            payload = json.loads(session_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping legacy session file %s: %s", session_file, exc)
            continue
        if not isinstance(payload, dict):
            continue
        messages = payload.get("messages")
        if not isinstance(messages, list):
            logger.warning("Skipping legacy session %s: missing messages list", session_file)
            continue
        raw_name = payload.get("name")
        name = raw_name if isinstance(raw_name, str) and raw_name else session_file.stem
        raw_saved_at = payload.get("saved_at")
        saved_at = (
            raw_saved_at
            if isinstance(raw_saved_at, str) and raw_saved_at
            else datetime.fromtimestamp(session_file.stat().st_mtime).isoformat()
        )
        conn.execute(
            "INSERT OR IGNORE INTO sessions (name, saved_at, message_count, messages) "
            "VALUES (?, ?, ?, ?)",
            (name, saved_at, len(messages), json.dumps(messages)),
        )
        imported += 1
    if imported:
        logger.info("Migrated %d session(s) from %s", imported, path)


def _import_index(conn: sqlite3.Connection, path: Path) -> None:
    if not path.exists() or _count(conn, "index_roots") > 0:
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Skipping legacy index file %s: %s", path, exc)
        return
    if not isinstance(payload, dict):
        return
    imported = 0
    for root, info in payload.items():
        if not isinstance(info, dict):
            continue
        conn.execute(
            "INSERT OR IGNORE INTO index_roots "
            "(root, file_count, dir_count, total_size, indexed_at, entries) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                root,
                info.get("file_count", 0),
                info.get("dir_count", 0),
                info.get("total_size", 0),
                info.get("indexed_at", ""),
                json.dumps(info.get("entries", {})),
            ),
        )
        imported += 1
    if imported:
        logger.info("Migrated %d indexed root(s) from %s", imported, path)


def _count(conn: sqlite3.Connection, table: str) -> int:
    row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    return int(row[0]) if row is not None else 0
