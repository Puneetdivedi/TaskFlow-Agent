"""Task-management tool — lets the agent maintain the user's task list."""

from __future__ import annotations

import asyncio
from typing import Any

from src.interfaces.task_store import TASK_PRIORITIES, TASK_STATUSES, ITaskStore, Task, TaskListOptions
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
            "description, priority, due_at, every_days recurrence, plan, and auto_run), "
            "'list' tasks (optionally filtered by status, priority, due date range), "
            "'search' tasks with advanced filters (keyword, status, priority, due_before, "
            "due_after, sort, pagination), 'get' one task's details, "
            "'update' a task's title/description/status/priority/due_at/every_days/plan/"
            "auto_run, 'complete' a task (recurring tasks roll their due date forward), "
            "'due' list tasks due now or within ahead_days, or 'delete' a task. A task "
            "with a 'plan' can be executed autonomously with the automation_run tool."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "list", "get", "update", "complete", "due", "delete", "search"],
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
                "due_at": {
                    "type": "string",
                    "description": (
                        "ISO-8601 due date/time (e.g. '2026-08-10' or '2026-08-10T09:30') "
                        "— empty string clears it"
                    ),
                },
                "every_days": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Recurrence interval in days (0 = not recurring)",
                },
                "plan": {
                    "type": "string",
                    "description": (
                        "Multi-step instructions for autonomous execution — plain text, "
                        "one step per line. Empty string clears it."
                    ),
                },
                "auto_run": {
                    "type": "boolean",
                    "description": (
                        "Run the task's plan autonomously when it comes due (requires "
                        "a non-empty 'plan')."
                    ),
                },
                "ahead_days": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Horizon in days for the 'due' action (default: 0)",
                },
                "due_before": {
                    "type": "string",
                    "description": "ISO-8601 date/time - filter tasks due at or before this (search action)",
                },
                "due_after": {
                    "type": "string",
                    "description": "ISO-8601 date/time - filter tasks due at or after this (search action)",
                },
                "search": {
                    "type": "string",
                    "description": "Keyword search in title and description (search action)",
                },
                "sort_by": {
                    "type": "string",
                    "enum": ["created_at", "updated_at", "due_at", "priority", "title"],
                    "description": "Sort field for search action (default: created_at)",
                },
                "sort_desc": {
                    "type": "boolean",
                    "description": "Sort descending (newest/soonest first) for search action (default: true)",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Max results for search action",
                },
                "offset": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Pagination offset for search action (default: 0)",
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
                    kwargs.get("due_at", ""),
                    kwargs.get("every_days", 0),
                    kwargs.get("plan", ""),
                    bool(kwargs.get("auto_run", False)),
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

        if action == "search":
            try:
                options = TaskListOptions(
                    status=kwargs.get("status"),
                    priority=kwargs.get("priority"),
                    due_before=kwargs.get("due_before"),
                    due_after=kwargs.get("due_after"),
                    search=kwargs.get("search"),
                    sort_by=kwargs.get("sort_by", "created_at"),
                    sort_desc=kwargs.get("sort_desc", True),
                    limit=kwargs.get("limit"),
                    offset=kwargs.get("offset", 0),
                )
                tasks = await asyncio.to_thread(self._store.search, options)
            except ValueError as exc:
                raise ToolError(str(exc)) from exc
            return self._format_list(tasks)

        if action == "get":
            return await asyncio.to_thread(self._get, kwargs.get("task_id", ""))

        if action == "update":
            return await asyncio.to_thread(self._update, kwargs.get("task_id", ""), kwargs)

        if action == "complete":
            return await asyncio.to_thread(self._complete, kwargs.get("task_id", ""))

        if action == "due":
            try:
                tasks = await asyncio.to_thread(self._store.due, int(kwargs.get("ahead_days", 0)))
            except (TypeError, ValueError) as exc:
                raise ToolError(str(exc)) from exc
            return self._format_list(tasks)

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
        auto_run = changes.get("auto_run")
        try:
            task = self._store.update(
                task_id,
                title=changes.get("title"),
                description=changes.get("description"),
                status=changes.get("status"),
                priority=changes.get("priority"),
                due_at=changes.get("due_at"),
                every_days=changes.get("every_days"),
                plan=changes.get("plan"),
                auto_run=None if auto_run is None else bool(auto_run),
            )
        except (KeyError, ValueError) as exc:
            raise ToolError(str(exc)) from exc
        return (
            f"Updated task {task.id}: {task.title} "
            f"(status: {task.status}, priority: {task.priority})"
        )

    def _complete(self, task_id: str) -> str:
        try:
            task = self._store.complete(task_id)
        except (KeyError, ValueError) as exc:
            raise ToolError(str(exc)) from exc
        if task.every_days > 0 and task.due_at:
            return f"Completed task {task.id}: {task.title} (next due: {task.due_at[:10]})"
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
        if task.due_at:
            lines.append(f"  due: {task.due_at[:10]}")
        if task.every_days:
            lines.append(f"  repeats every {task.every_days} day(s)")
        if task.auto_run:
            lines.append("  ⚙  auto-runs when due")
        if task.plan:
            lines.append(f"  plan: {task.plan}")
        lines.append(f"  created: {task.created_at[:19]}")
        return "\n".join(lines)

    @staticmethod
    def _format_list(tasks: list[Task]) -> str:
        if not tasks:
            return "No tasks."
        lines = [f"{len(tasks)} task(s):"]
        for task in tasks:
            due = f", due: {task.due_at[:10]}" if task.due_at else ""
            auto = ", auto-run" if task.auto_run else ""
            lines.append(
                f"  [{task.status:<11}] {task.id} {task.title} "
                f"(priority: {task.priority}, created: {task.created_at[:10]}{due}{auto})"
            )
        return "\n".join(lines)
