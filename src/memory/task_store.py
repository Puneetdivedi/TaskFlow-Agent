"""Persistent task storage — a SQLite-backed task list.

Tasks live in the ``tasks`` table of the shared TaskFlow database
(default ``~/.taskflow/taskflow.db``), mirroring the ``SessionStore`` and
``FileIndex`` persistence pattern.
"""

from __future__ import annotations

import logging
import re
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from src.interfaces.task_store import TASK_PRIORITIES, TASK_STATUSES, Task
from src.memory.sqlite_store import TASKS_SCHEMA, SQLiteStore

logger = logging.getLogger(__name__)

_ID_RE = re.compile(r"^t(\d+)$")

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
    "plan",
    "auto_run",
)

#: Columns added after the original table was created. ``CREATE TABLE IF NOT
#: EXISTS`` alone never alters an existing table, so each new column is applied
#: with ``ALTER TABLE ... ADD COLUMN`` in :meth:`TaskStore._ensure_columns`.
_ADDED_COLUMNS: tuple[tuple[str, str], ...] = (
    ("plan", "TEXT NOT NULL DEFAULT ''"),
    ("auto_run", "INTEGER NOT NULL DEFAULT 0"),
)


def _row_to_task(row: Any) -> Task:
    """Build a :class:`Task` from a sqlite row, normalizing ``auto_run``.

    The column is stored as an ``INTEGER`` (0/1); coercing it to ``bool`` keeps
    the dataclass field's advertised type honest.
    """
    data = dict(row)
    data["auto_run"] = bool(data.get("auto_run", 0))
    return Task(**data)


class TaskStore(SQLiteStore):
    """SQLite-backed store for a persistent task list."""

    _SCHEMA = TASKS_SCHEMA

    def __init__(self, db_path: Path | None = None) -> None:
        super().__init__(db_path)
        self._ensure_columns()

    def _ensure_columns(self) -> None:
        """Add any new ``tasks`` columns that a pre-existing database lacks."""
        with self._connect() as conn:
            existing = {row["name"] for row in conn.execute("PRAGMA table_info(tasks)")}
        missing = [(name, ddl) for name, ddl in _ADDED_COLUMNS if name not in existing]
        if not missing:
            return
        with self._connect() as conn:
            for name, ddl in missing:
                conn.execute(f"ALTER TABLE tasks ADD COLUMN {name} {ddl}")
        logger.info("Migrated tasks table: added column(s) %s", ", ".join(n for n, _ in missing))

    # ------------------------------------------------------------------
    def _next_id(self) -> str:
        with self._connect() as conn:
            rows = conn.execute("SELECT id FROM tasks").fetchall()
        max_n = 0
        for row in rows:
            match = _ID_RE.match(row["id"])
            if match:
                max_n = max(max_n, int(match.group(1)))
        return f"t{max_n + 1}"

    @staticmethod
    def _now() -> str:
        return datetime.now().isoformat()

    @staticmethod
    def _validate_title(title: str) -> None:
        if not title.strip():
            raise ValueError("Task title must not be empty")

    @staticmethod
    def _validate_status(status: str) -> None:
        if status not in TASK_STATUSES:
            raise ValueError(
                f"Invalid status {status!r} — must be one of {', '.join(TASK_STATUSES)}"
            )

    @staticmethod
    def _validate_priority(priority: str) -> None:
        if priority not in TASK_PRIORITIES:
            raise ValueError(
                f"Invalid priority {priority!r} — must be one of {', '.join(TASK_PRIORITIES)}"
            )

    @staticmethod
    def _validate_due_at(due_at: str) -> None:
        """Empty is allowed; otherwise must be naive ISO-8601 (no tz offset)."""
        if not due_at:
            return
        try:
            parsed = datetime.fromisoformat(due_at)
        except ValueError:
            raise ValueError(
                f"Invalid due_at {due_at!r} — expected ISO-8601 "
                "(e.g. '2026-08-10' or '2026-08-10T09:30')"
            ) from None
        if parsed.tzinfo is not None:
            raise ValueError("due_at must not include a timezone offset")

    @staticmethod
    def _validate_every_days(every_days: int) -> None:
        """Reject negative values (and bools, a subclass of int)."""
        if type(every_days) is not int or every_days < 0:
            raise ValueError(
                f"Invalid every_days {every_days!r} — must be a non-negative "
                "integer (0 = not recurring)"
            )

    def _find(self, task_id: str) -> Task:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT {', '.join(_TASK_COLUMNS)} FROM tasks WHERE id = ?",
                (task_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Task {task_id!r} not found")
        return _row_to_task(row)

    def _save_row(self, task: Task) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE tasks SET title = ?, description = ?, status = ?, priority = ?, "
                "created_at = ?, updated_at = ?, due_at = ?, every_days = ?, plan = ?, "
                "auto_run = ? WHERE id = ?",
                (
                    task.title,
                    task.description,
                    task.status,
                    task.priority,
                    task.created_at,
                    task.updated_at,
                    task.due_at,
                    task.every_days,
                    task.plan,
                    int(task.auto_run),
                    task.id,
                ),
            )

    # ------------------------------------------------------------------
    def create(
        self,
        title: str,
        description: str = "",
        priority: str = "medium",
        due_at: str = "",
        every_days: int = 0,
        plan: str = "",
        auto_run: bool = False,
    ) -> Task:
        """Create a task and persist it. Returns the new task."""
        self._validate_title(title)
        self._validate_priority(priority)
        self._validate_due_at(due_at)
        self._validate_every_days(every_days)
        now = self._now()
        task = Task(
            id=self._next_id(),
            title=title.strip(),
            description=description,
            status="todo",
            priority=priority,
            created_at=now,
            updated_at=now,
            due_at=due_at,
            every_days=every_days,
            plan=plan,
            auto_run=auto_run,
        )
        with self._connect() as conn:
            conn.execute(
                f"INSERT INTO tasks ({', '.join(_TASK_COLUMNS)}) "
                f"VALUES ({', '.join('?' * len(_TASK_COLUMNS))})",
                tuple(task.__dict__[col] for col in _TASK_COLUMNS),
            )
        logger.info("Created task %s: %s", task.id, task.title)
        return task

    def get(self, task_id: str) -> Task:
        """Return the task with *task_id*.

        Raises ``KeyError`` if no such task exists.
        """
        return self._find(task_id)

    def due(self, ahead_days: int = 0) -> list[Task]:
        """Return non-done tasks due at or before now + *ahead_days*,
        soonest first. Tasks with an unparseable ``due_at`` are skipped."""
        if ahead_days < 0:
            raise ValueError(f"ahead_days must be >= 0, got {ahead_days}")
        cutoff = datetime.now() + timedelta(days=ahead_days)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT "
                + ", ".join(_TASK_COLUMNS)
                + " FROM tasks WHERE status != 'done' AND due_at != ''"
            ).fetchall()
        tasks = [_row_to_task(row) for row in rows]
        due_tasks: list[tuple[datetime, Task]] = []
        for task in tasks:
            try:
                due = datetime.fromisoformat(task.due_at)
            except ValueError:
                logger.warning("Skipping task %s with unparseable due_at %r", task.id, task.due_at)
                continue
            if due <= cutoff:
                due_tasks.append((due, task))
        due_tasks.sort(key=lambda pair: pair[0])
        return [task for _, task in due_tasks]

    def list(self, status: str | None = None) -> list[Task]:
        """Return all tasks (optionally filtered by *status*), newest first."""
        with self._connect() as conn:
            if status is not None:
                self._validate_status(status)
                rows = conn.execute(
                    "SELECT "
                    + ", ".join(_TASK_COLUMNS)
                    + " FROM tasks WHERE status = ? ORDER BY created_at DESC",
                    (status,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT " + ", ".join(_TASK_COLUMNS) + " FROM tasks ORDER BY created_at DESC"
                ).fetchall()
        return [_row_to_task(row) for row in rows]

    def update(
        self,
        task_id: str,
        *,
        title: str | None = None,
        description: str | None = None,
        status: str | None = None,
        priority: str | None = None,
        due_at: str | None = None,
        every_days: int | None = None,
        plan: str | None = None,
        auto_run: bool | None = None,
    ) -> Task:
        """Update the given fields of *task_id* and persist the change.

        Raises ``KeyError`` if no such task exists and ``ValueError`` for
        invalid field values.
        """
        task = self._find(task_id)
        changes: dict[str, Any] = {}
        if title is not None:
            self._validate_title(title)
            changes["title"] = title.strip()
        if description is not None:
            changes["description"] = description
        if status is not None:
            self._validate_status(status)
            changes["status"] = status
        if priority is not None:
            self._validate_priority(priority)
            changes["priority"] = priority
        if due_at is not None:
            self._validate_due_at(due_at)
            changes["due_at"] = due_at
        if every_days is not None:
            self._validate_every_days(every_days)
            changes["every_days"] = every_days
        if plan is not None:
            changes["plan"] = plan
        if auto_run is not None:
            changes["auto_run"] = auto_run
        if not changes:
            return task

        updated = replace(task, **changes, updated_at=self._now())
        fields = {**changes, "updated_at": updated.updated_at}
        sets = ", ".join(f"{name} = ?" for name in fields)
        values = list(fields.values()) + [task_id]
        with self._connect() as conn:
            conn.execute(f"UPDATE tasks SET {sets} WHERE id = ?", values)
        logger.info("Updated task %s (fields: %s)", task_id, ", ".join(changes))
        return updated

    def complete(self, task_id: str) -> Task:
        """Mark a task done; recurring tasks (``every_days > 0`` with a
        parseable ``due_at``) roll their due date forward by ``every_days``
        and stay ``todo`` instead."""
        task = self._find(task_id)
        now = self._now()
        if task.every_days > 0 and task.due_at:
            try:
                due = datetime.fromisoformat(task.due_at)
            except ValueError:
                logger.warning(
                    "Task %s has unparseable due_at %r — marking done",
                    task_id,
                    task.due_at,
                )
            else:
                rolled = due + timedelta(days=task.every_days)
                # Preserve a bare-date input as bare-date output, so
                # `due_at=2026-08-10` rolls to `2026-08-17`, not `...T00:00:00`.
                next_due = (
                    rolled.date().isoformat() if len(task.due_at) == 10 else rolled.isoformat()
                )
                updated = replace(task, status="todo", due_at=next_due, updated_at=now)
                self._save_row(updated)
                logger.info("Completed recurring task %s — next due %s", task_id, next_due)
                return updated
        updated = replace(task, status="done", updated_at=now)
        self._save_row(updated)
        logger.info("Completed task %s", task_id)
        return updated

    def advance(self, task_id: str) -> Task:
        """Roll a recurring task's ``due_at`` forward by ``every_days``.

        Unlike :meth:`complete`, the task's status is preserved — an
        autonomous run finished it, but the recurring cycle continues. One-shot
        tasks (``every_days == 0``) are marked ``done``. Raises ``KeyError`` if
        no such task exists.
        """
        task = self._find(task_id)
        now = self._now()
        if task.every_days > 0 and task.due_at:
            try:
                due = datetime.fromisoformat(task.due_at)
            except ValueError:
                logger.warning(
                    "Task %s has unparseable due_at %r — marking done",
                    task_id,
                    task.due_at,
                )
            else:
                rolled = due + timedelta(days=task.every_days)
                next_due = (
                    rolled.date().isoformat() if len(task.due_at) == 10 else rolled.isoformat()
                )
                updated = replace(task, due_at=next_due, updated_at=now)
                self._save_row(updated)
                logger.info("Advanced recurring task %s — next due %s", task_id, next_due)
                return updated
        updated = replace(task, status="done", updated_at=now)
        self._save_row(updated)
        logger.info("Advanced one-shot task %s (marked done)", task_id)
        return updated

    def delete(self, task_id: str) -> None:
        """Remove the task with *task_id*.

        Raises ``KeyError`` if no such task exists.
        """
        self._find(task_id)
        with self._connect() as conn:
            conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        logger.info("Deleted task %s", task_id)
