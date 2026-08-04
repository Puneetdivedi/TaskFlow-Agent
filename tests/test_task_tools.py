"""Tests for the task-management tool."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from src.interfaces.task_store import TASK_PRIORITIES, TASK_STATUSES
from src.memory.task_store import TaskStore
from src.tools.base import ToolError
from src.tools.registry import ToolRegistry
from src.tools.task_tools import TaskTool


@pytest.fixture
def tool(task_store: TaskStore) -> TaskTool:
    return TaskTool(task_store)


class TestTaskToolContract:
    def test_name_is_tasks(self, tool: TaskTool) -> None:
        assert tool.name == "tasks"

    def test_description_mentions_actions(self, tool: TaskTool) -> None:
        assert "create" in tool.description
        assert "delete" in tool.description

    def test_input_schema_requires_action(self, tool: TaskTool) -> None:
        schema = tool.input_schema
        assert schema["required"] == ["action"]
        assert "create" in schema["properties"]["action"]["enum"]
        assert schema["properties"]["priority"]["enum"] == list(TASK_PRIORITIES)
        assert schema["properties"]["status"]["enum"] == list(TASK_STATUSES)

    def test_input_schema_includes_scheduling(self, tool: TaskTool) -> None:
        schema = tool.input_schema
        assert "due" in schema["properties"]["action"]["enum"]
        assert "due_at" in schema["properties"]
        assert "every_days" in schema["properties"]
        assert "ahead_days" in schema["properties"]


class TestTaskToolCreate:
    async def test_create_creates_task(self, tool: TaskTool, task_store: TaskStore) -> None:
        result = await tool.run(action="create", title="Buy milk")
        assert result == "Created task t1: Buy milk (status: todo, priority: medium)"
        assert task_store.get("t1").title == "Buy milk"

    async def test_create_rejects_empty_title(self, tool: TaskTool) -> None:
        with pytest.raises(ToolError, match="must not be empty"):
            await tool.run(action="create", title="")

    async def test_create_rejects_bad_priority(self, tool: TaskTool) -> None:
        with pytest.raises(ToolError, match="Invalid priority"):
            await tool.run(action="create", title="Task", priority="urgent")

    async def test_create_with_due_at_and_every_days(
        self, tool: TaskTool, task_store: TaskStore
    ) -> None:
        result = await tool.run(
            action="create", title="Water plants", due_at="2026-08-10", every_days=7
        )
        assert "Created task t1" in result
        assert task_store.get("t1").due_at == "2026-08-10"
        assert task_store.get("t1").every_days == 7

    async def test_create_rejects_bad_due_at(self, tool: TaskTool) -> None:
        with pytest.raises(ToolError, match="Invalid due_at"):
            await tool.run(action="create", title="Task", due_at="bogus")


class TestTaskToolList:
    async def test_list_empty(self, tool: TaskTool) -> None:
        assert (await tool.run(action="list")) == "No tasks."

    async def test_list_returns_tasks(self, tool: TaskTool) -> None:
        await tool.run(action="create", title="First")
        await tool.run(action="create", title="Second")
        result = await tool.run(action="list")
        assert "2 task(s):" in result
        assert "t1 First" in result
        assert "t2 Second" in result

    async def test_list_filters_by_status(self, tool: TaskTool, task_store: TaskStore) -> None:
        await tool.run(action="create", title="First")
        await tool.run(action="complete", task_id="t1")
        result = await tool.run(action="list", status="done")
        assert "t1" in result
        assert "t2" not in result

    async def test_list_invalid_status_raises(self, tool: TaskTool) -> None:
        with pytest.raises(ToolError, match="Invalid status"):
            await tool.run(action="list", status="bogus")


class TestTaskToolGet:
    async def test_get_returns_task(self, tool: TaskTool, task_store: TaskStore) -> None:
        task_store.create("Buy milk")
        result = await tool.run(action="get", task_id="t1")
        assert "t1: Buy milk" in result
        assert "status: todo" in result
        assert "priority: medium" in result

    async def test_get_missing_raises(self, tool: TaskTool) -> None:
        with pytest.raises(ToolError, match="not found"):
            await tool.run(action="get", task_id="t99")


class TestTaskToolUpdate:
    async def test_update_changes_fields(self, tool: TaskTool, task_store: TaskStore) -> None:
        task_store.create("Buy milk")
        result = await tool.run(action="update", task_id="t1", status="done", priority="high")
        assert "Updated task t1: Buy milk (status: done, priority: high)" in result
        assert task_store.get("t1").status == "done"
        assert task_store.get("t1").priority == "high"

    async def test_update_missing_raises(self, tool: TaskTool) -> None:
        with pytest.raises(ToolError, match="not found"):
            await tool.run(action="update", task_id="t99", status="done")

    async def test_update_invalid_status_raises(
        self, tool: TaskTool, task_store: TaskStore
    ) -> None:
        task_store.create("Buy milk")
        with pytest.raises(ToolError, match="Invalid status"):
            await tool.run(action="update", task_id="t1", status="bogus")

    async def test_update_due_at_and_every_days(
        self, tool: TaskTool, task_store: TaskStore
    ) -> None:
        task_store.create("Task")
        result = await tool.run(action="update", task_id="t1", due_at="2026-08-10", every_days=7)
        assert "Updated task t1" in result
        assert task_store.get("t1").due_at == "2026-08-10"
        assert task_store.get("t1").every_days == 7


class TestTaskToolComplete:
    async def test_complete_marks_done(self, tool: TaskTool, task_store: TaskStore) -> None:
        task_store.create("Buy milk")
        result = await tool.run(action="complete", task_id="t1")
        assert result == "Completed task t1: Buy milk"
        assert task_store.get("t1").status == "done"

    async def test_complete_recurring_returns_next_due(
        self, tool: TaskTool, task_store: TaskStore
    ) -> None:
        task_store.create("Water plants", due_at="2026-08-10", every_days=7)
        result = await tool.run(action="complete", task_id="t1")
        assert "(next due: 2026-08-17)" in result
        assert task_store.get("t1").status == "todo"
        assert task_store.get("t1").due_at == "2026-08-17"

    async def test_complete_missing_raises(self, tool: TaskTool) -> None:
        with pytest.raises(ToolError, match="not found"):
            await tool.run(action="complete", task_id="t99")


class TestTaskToolDue:
    async def test_due_empty(self, tool: TaskTool) -> None:
        assert (await tool.run(action="due")) == "No tasks."

    async def test_due_returns_due_tasks(self, tool: TaskTool, task_store: TaskStore) -> None:
        due_date = (datetime.now() - timedelta(days=1)).date().isoformat()
        task_store.create("Overdue", due_at=due_date)
        result = await tool.run(action="due")
        assert "t1" in result
        assert "Overdue" in result

    async def test_due_with_ahead_days(self, tool: TaskTool, task_store: TaskStore) -> None:
        due_date = (datetime.now() + timedelta(days=3)).date().isoformat()
        task_store.create("Later", due_at=due_date)
        assert (await tool.run(action="due")) == "No tasks."
        result = await tool.run(action="due", ahead_days=7)
        assert "Later" in result


class TestTaskToolDelete:
    async def test_delete_removes_task(self, tool: TaskTool, task_store: TaskStore) -> None:
        task_store.create("Buy milk")
        result = await tool.run(action="delete", task_id="t1")
        assert result == "Deleted task t1"
        with pytest.raises(KeyError):
            task_store.get("t1")

    async def test_delete_missing_raises(self, tool: TaskTool) -> None:
        with pytest.raises(ToolError, match="not found"):
            await tool.run(action="delete", task_id="t99")


class TestTaskToolMisc:
    async def test_unknown_action_raises(self, tool: TaskTool) -> None:
        with pytest.raises(ToolError, match="Unknown tasks action"):
            await tool.run(action="frobnicate")


class TestTaskToolRegistryIntegration:
    def test_registry_includes_tasks_with_store(self, task_store: TaskStore) -> None:
        registry = ToolRegistry(task_store=task_store)
        assert "tasks" in registry.tool_names

    def test_registry_excludes_tasks_without_store(self) -> None:
        registry = ToolRegistry()
        assert "tasks" not in registry.tool_names
