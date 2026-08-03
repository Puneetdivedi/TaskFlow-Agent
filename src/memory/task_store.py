"""Persistent task storage — a JSON file holding the user's task list.

Tasks live in a single ``tasks.json`` file (default ``~/.taskflow/tasks.json``),
mirroring the ``SessionStore``/``FileIndex`` JSON persistence pattern.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import replace
from datetime import datetime
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
    ) -> Task:
        """Create a task and persist it. Returns the new task."""
        self._validate_title(title)
        self._validate_priority(priority)
        now = self._now()
        task = Task(
            id=self._next_id(),
            title=title.strip(),
            description=description,
            status="todo",
            priority=priority,
            created_at=now,
            updated_at=now,
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
        if not changes:
            return task

        updated = replace(task, **changes, updated_at=self._now())
        self._tasks[self._tasks.index(task)] = updated
        self._save()
        logger.info("Updated task %s (fields: %s)", task_id, ", ".join(changes))
        return updated

    def delete(self, task_id: str) -> None:
        """Remove the task with *task_id*.

        Raises ``KeyError`` if no such task exists.
        """
        task = self._find(task_id)
        self._tasks.remove(task)
        self._save()
        logger.info("Deleted task %s", task_id)
