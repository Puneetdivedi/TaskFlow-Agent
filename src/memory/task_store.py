"""Persistent task storage — a JSON file holding the user's task list.

Tasks live in a single ``tasks.json`` file (default ``~/.taskflow/tasks.json``),
mirroring the ``SessionStore``/``FileIndex`` JSON persistence pattern.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from src.interfaces.task_store import TASK_PRIORITIES, TASK_STATUSES, Task

logger = logging.getLogger(__name__)

_ID_RE = re.compile(r"^t(\d+)$")


class TaskStore:
    """JSON-backed store for a persistent task list."""

    def __init__(self, tasks_file: Path | None = None) -> None:
        self._file = tasks_file or Path.home() / ".taskflow" / "tasks.json"
        self._file.parent.mkdir(parents=True, exist_ok=True)
        self._tasks: list[Task] = []
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        if not self._file.exists():
            return
        try:
            payload = json.loads(self._file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Task store corrupted — starting empty: %s (%s)", self._file, exc)
            self._tasks = []
            return

        raw_tasks = payload.get("tasks", []) if isinstance(payload, dict) else []
        tasks: list[Task] = []
        for entry in raw_tasks:
            if not isinstance(entry, dict):
                continue
            try:
                tasks.append(Task(**entry))
            except (TypeError, ValueError):
                logger.warning("Skipping malformed task entry: %s", entry)
                continue
        self._tasks = tasks

    def _save(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        payload = {"tasks": [t.__dict__ for t in self._tasks]}
        self._file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    def _next_id(self) -> str:
        max_n = 0
        for task in self._tasks:
            match = _ID_RE.match(task.id)
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
        for task in self._tasks:
            if task.id == task_id:
                return task
        raise KeyError(f"Task {task_id!r} not found")

    # ------------------------------------------------------------------
    def create(
        self,
        title: str,
        description: str = "",
        priority: str = "medium",
        due_at: str = "",
        every_days: int = 0,
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
        )
        self._tasks.append(task)
        self._save()
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
        due_tasks: list[tuple[datetime, Task]] = []
        for task in self._tasks:
            if task.status == "done" or not task.due_at:
                continue
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
        if status is not None:
            self._validate_status(status)
            tasks = [t for t in self._tasks if t.status == status]
        else:
            tasks = list(self._tasks)
        return sorted(tasks, key=lambda t: t.created_at, reverse=True)

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
        if not changes:
            return task

        updated = replace(task, **changes, updated_at=self._now())
        self._tasks[self._tasks.index(task)] = updated
        self._save()
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
                self._tasks[self._tasks.index(task)] = updated
                self._save()
                logger.info("Completed recurring task %s — next due %s", task_id, next_due)
                return updated
        updated = replace(task, status="done", updated_at=now)
        self._tasks[self._tasks.index(task)] = updated
        self._save()
        logger.info("Completed task %s", task_id)
        return updated

    def delete(self, task_id: str) -> None:
        """Remove the task with *task_id*.

        Raises ``KeyError`` if no such task exists.
        """
        task = self._find(task_id)
        self._tasks.remove(task)
        self._save()
        logger.info("Deleted task %s", task_id)
