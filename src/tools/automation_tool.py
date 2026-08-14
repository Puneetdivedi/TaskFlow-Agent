"""Tool that executes a task's stored plan autonomously."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from src.agent.automation import AutomationRunner
from src.interfaces import ITaskStore
from src.tools.base import Tool, ToolError

logger = logging.getLogger(__name__)


class AutomationRunTool(Tool):
    """Run a task's ``plan`` to completion in an isolated, bounded loop.

    Loads the task, refuses to run when it has no plan, executes the plan via
    :class:`AutomationRunner`, then advances the task (recurring tasks roll
    their due date forward; one-shots are marked done).
    """

    def __init__(self, runner: AutomationRunner, store: ITaskStore) -> None:
        self._runner = runner
        self._store = store

    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return "automation_run"

    @property
    def description(self) -> str:
        return (
            "Execute a task's stored plan autonomously, from start to finish, in an "
            "isolated tool loop. Pass the task_id of a task that has a plan "
            "(create one with the 'tasks' tool's plan field). After the run the task "
            "is advanced: recurring tasks roll their due date forward, one-shots are "
            "marked done."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "Task id (e.g. 't3') whose plan should be run.",
                },
            },
            "required": ["task_id"],
        }

    # ------------------------------------------------------------------
    async def run(self, task_id: str, **kwargs: Any) -> str:  # type: ignore[override]
        try:
            task = await asyncio.to_thread(self._store.get, task_id)
        except KeyError as exc:
            raise ToolError(str(exc)) from exc
        if not task.plan:
            raise ToolError(
                f"Task {task_id} has no plan — add one with the 'tasks' tool's "
                "update action (plan=...)."
            )
        logger.info("Running plan for task %s: %s", task_id, task.title)
        report = await self._runner.run_plan(task.plan, title=task.title)
        await asyncio.to_thread(self._store.advance, task_id)
        return f"Ran plan for {task_id}: {task.title}\n\n{report}"
