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


class ITaskStore(Protocol):
    """Interface for a persistent, ordered list of tasks."""

    def create(
        self,
        title: str,
        description: str = "",
        priority: str = "medium",
    ) -> Task:
        """Create a task and persist it. Returns the new task."""
        ...

    def get(self, task_id: str) -> Task:
        """Return the task with *task_id*.

        Raises ``KeyError`` if no such task exists.
        """
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
    ) -> Task:
        """Update the given fields of *task_id* and persist the change.

        Raises ``KeyError`` if no such task exists and ``ValueError`` for
        invalid field values.
        """
        ...

    def delete(self, task_id: str) -> None:
        """Remove the task with *task_id*.

        Raises ``KeyError`` if no such task exists.
        """
        ...
