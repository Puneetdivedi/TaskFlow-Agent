"""Task-management tool — lets the agent maintain the user's task list."""

from __future__ import annotations

import asyncio
from typing import Any

from src.interfaces.task_store import TASK_PRIORITIES, TASK_STATUSES, ITaskStore, Task
from src.tools.base import Tool, ToolError


class TaskTool(Tool):
    """A single ``tasks`` tool with an ``action`` field (like ``file_index``)."""

    def __init__(self, store: ITaskStore) -> None:
        self._store = store

    @property
    def name(self) -> str:
        return "tasks"

    @property
    def description(self) -> str:
        return (
            "Manage a persistent task list. Actions: 'create' a task (title, plus optional "
            "description and priority), 'list' tasks (optionally filtered by status), "
            "'get' one task's details, 'update' a task's title/description/status/priority, "
            "'complete' a task, or 'delete' a task."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "list", "get", "update", "complete", "delete"],
                    "description": "What to do with the task list",
                },
                "task_id": {
                    "type": "string",
                    "description": "Task id (e.g. 't1') — for get/update/complete/delete",
                },
                "title": {"type": "string", "description": "Task title (required for create)"},
                "description": {"type": "string", "description": "Optional task description"},
                "priority": {
                    "type": "string",
                    "enum": list(TASK_PRIORITIES),
                    "description": "Priority (default: medium)",
                },
                "status": {
                    "type": "string",
                    "enum": list(TASK_STATUSES),
                    "description": "Status filter for 'list', or the new status for 'update'",
                },
            },
            "required": ["action"],
        }

    # ------------------------------------------------------------------
    async def run(self, action: str, **kwargs: Any) -> str:  # type: ignore[override]
        if action == "create":
            try:
                task = await asyncio.to_thread(
                    self._store.create,
                    kwargs.get("title", ""),
                    kwargs.get("description", ""),
                    kwargs.get("priority", "medium"),
                )
            except ValueError as exc:
                raise ToolError(str(exc)) from exc
            return (
                f"Created task {task.id}: {task.title} "
                f"(status: {task.status}, priority: {task.priority})"
            )

        if action == "list":
            status = kwargs.get("status")
            try:
                tasks = await asyncio.to_thread(self._store.list, status)
            except ValueError as exc:
                raise ToolError(str(exc)) from exc
            return self._format_list(tasks)

        if action == "get":
            return await asyncio.to_thread(self._get, kwargs.get("task_id", ""))

        if action == "update":
            return await asyncio.to_thread(self._update, kwargs.get("task_id", ""), kwargs)

        if action == "complete":
            return await asyncio.to_thread(self._complete, kwargs.get("task_id", ""))

        if action == "delete":
            return await asyncio.to_thread(self._delete, kwargs.get("task_id", ""))

        raise ToolError(f"Unknown tasks action: {action!r}")

    # ------------------------------------------------------------------
    def _get(self, task_id: str) -> str:
        try:
            task = self._store.get(task_id)
        except KeyError as exc:
            raise ToolError(str(exc)) from exc
        return self._format_task(task)

    def _update(self, task_id: str, changes: dict[str, Any]) -> str:
        try:
            task = self._store.update(
                task_id,
                title=changes.get("title"),
                description=changes.get("description"),
                status=changes.get("status"),
                priority=changes.get("priority"),
            )
        except (KeyError, ValueError) as exc:
            raise ToolError(str(exc)) from exc
        return (
            f"Updated task {task.id}: {task.title} "
            f"(status: {task.status}, priority: {task.priority})"
        )

    def _complete(self, task_id: str) -> str:
        try:
            task = self._store.update(task_id, status="done")
        except (KeyError, ValueError) as exc:
            raise ToolError(str(exc)) from exc
        return f"Completed task {task.id}: {task.title}"

    def _delete(self, task_id: str) -> str:
        try:
            self._store.delete(task_id)
        except KeyError as exc:
            raise ToolError(str(exc)) from exc
        return f"Deleted task {task_id}"

    # ------------------------------------------------------------------
    @staticmethod
    def _format_task(task: Task) -> str:
        lines = [f"{task.id}: {task.title} [status: {task.status}, priority: {task.priority}]"]
        if task.description:
            lines.append(f"  {task.description}")
        lines.append(f"  created: {task.created_at[:19]}")
        return "\n".join(lines)

    @staticmethod
    def _format_list(tasks: list[Task]) -> str:
        if not tasks:
            return "No tasks."
        lines = [f"{len(tasks)} task(s):"]
        for task in tasks:
            lines.append(
                f"  [{task.status:<11}] {task.id} {task.title} "
                f"(priority: {task.priority}, created: {task.created_at[:10]})"
            )
        return "\n".join(lines)
