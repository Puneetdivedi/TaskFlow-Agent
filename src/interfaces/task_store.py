"""Protocol for persistent task storage (create/list/update/delete tasks)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

# Allowed statuses and priorities — shared by the store (validation)
# and the tool (descriptions).
TASK_STATUSES = ("todo", "in_progress", "done")
TASK_PRIORITIES = ("low", "medium", "high")


@dataclass(frozen=True)
class Task:
    """A single task in the user's persistent task list."""

    id: str
    title: str
    description: str = ""
    status: str = "todo"
    priority: str = "medium"
    created_at: str = ""
    updated_at: str = ""
    due_at: str = ""  # ISO-8601 due date/time; "" = no due date
    every_days: int = 0  # recurrence interval in days; 0 = not recurring


class ITaskStore(Protocol):
    """Interface for a persistent, ordered list of tasks."""

    def create(
        self,
        title: str,
        description: str = "",
        priority: str = "medium",
        due_at: str = "",
        every_days: int = 0,
    ) -> Task:
        """Create a task and persist it. Returns the new task."""
        ...

    def get(self, task_id: str) -> Task:
        """Return the task with *task_id*.

        Raises ``KeyError`` if no such task exists.
        """
        ...

    def due(self, ahead_days: int = 0) -> list[Task]:
        """Return non-done tasks due at or before now + *ahead_days*,
        soonest first."""
        ...

    def list(self, status: str | None = None) -> list[Task]:
        """Return all tasks (optionally filtered by *status*), newest first."""
        ...

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
        ...

    def complete(self, task_id: str) -> Task:
        """Mark a task done; recurring tasks (``every_days > 0`` with a
        ``due_at``) roll their due date forward by ``every_days`` and stay
        ``todo`` instead of being completed."""
        ...

    def delete(self, task_id: str) -> None:
        """Remove the task with *task_id*.

        Raises ``KeyError`` if no such task exists.
        """
        ...
