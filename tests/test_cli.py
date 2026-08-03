"""Tests for the CLI helper functions (non-interactive parts)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.memory.session_store import SessionStore
from src.memory.task_store import TaskStore
from src.ui import cli as cli_module
from src.ui.cli import (
    format_session_list,
    format_task,
    format_task_list,
    handle_session_command,
    handle_task_command,
    print_banner,
    print_help,
    print_tools,
)


@pytest.fixture
def store(tmp_path: Path) -> SessionStore:
    return SessionStore(session_dir=tmp_path)


class _OrchStub:
    """Minimal orchestrator stand-in exposing only what the CLI loop touches."""

    def __init__(self, memory) -> None:
        self.memory = memory


def _script_prompt(monkeypatch, answers: list[str]) -> None:
    it = iter(answers)
    monkeypatch.setattr(cli_module.Prompt, "ask", lambda *a, **k: next(it))


class TestCLIHelpers:
    def test_print_banner_does_not_raise(self) -> None:
        print_banner()

    def test_print_help_does_not_raise(self) -> None:
        print_help()

    def test_print_tools_lists_registered_tools(self, mock_tool_registry) -> None:
        # mock_tool_registry only exposes "mock_tool"; build a minimal
        # stand-in orchestrator to exercise print_tools.
        class _Orch:
            tools = mock_tool_registry

        print_tools(_Orch())  # type: ignore[arg-type]


class TestSessionCommandHandler:
    def test_status_without_session(self, store: SessionStore, mock_memory) -> None:
        result = handle_session_command("", store, mock_memory, None)
        assert result.session is None
        assert "No active session" in result.message

    def test_status_with_current_session(self, store: SessionStore, mock_memory) -> None:
        mock_memory.add_user("hi")
        result = handle_session_command("", store, mock_memory, "work")
        assert result.session == "work"
        assert "work" in result.message
        assert "1 message" in result.message

    def test_new_creates_empty_session(self, store: SessionStore, mock_memory) -> None:
        result = handle_session_command("new work", store, mock_memory, None)
        assert result.session == "work"
        assert "Started new session" in result.message
        assert store.exists("work")
        assert mock_memory.messages == []  # memory cleared

    def test_new_without_name_auto_names(self, store: SessionStore, mock_memory) -> None:
        result = handle_session_command("new", store, mock_memory, None)
        assert result.session is not None
        assert store.exists(result.session)

    def test_new_checkpoints_previous_session(self, store: SessionStore, mock_memory) -> None:
        mock_memory.add_user("hello")
        mock_memory.add_assistant("hi")
        result = handle_session_command("new next", store, mock_memory, "prev")
        assert result.session == "next"
        assert store.load("prev") == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]

    def test_new_existing_name_refused(self, store: SessionStore, mock_memory) -> None:
        store.save("work", [])
        result = handle_session_command("new work", store, mock_memory, None)
        assert result.session is None
        assert "already exists" in result.message

    def test_new_invalid_name_keeps_memory(self, store: SessionStore, mock_memory) -> None:
        mock_memory.add_user("hi")
        result = handle_session_command("new a/b", store, mock_memory, None)
        assert result.session is None
        assert "must contain only" in result.message
        assert len(mock_memory.messages) == 1  # memory not cleared

    def test_save(self, store: SessionStore, mock_memory) -> None:
        mock_memory.add_user("hello")
        result = handle_session_command("save work", store, mock_memory, None)
        assert result.session is None  # current session unchanged
        assert "Saved session 'work'" in result.message
        assert store.load("work") == [{"role": "user", "content": "hello"}]

    def test_save_empty_aborts(self, store: SessionStore, mock_memory) -> None:
        result = handle_session_command("save work", store, mock_memory, None)
        assert "Nothing to save" in result.message
        assert not store.exists("work")

    def test_save_defaults_to_current_name(self, store: SessionStore, mock_memory) -> None:
        mock_memory.add_user("hi")
        result = handle_session_command("save", store, mock_memory, "work")
        assert "Saved session 'work'" in result.message
        assert store.exists("work")

    def test_load(self, store: SessionStore, mock_memory) -> None:
        saved = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        store.save("work", saved)
        result = handle_session_command("load work", store, mock_memory, None)
        assert result.session == "work"
        assert "Loaded session 'work'" in result.message
        assert mock_memory.messages == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

    def test_load_checkpoints_current(self, store: SessionStore, mock_memory) -> None:
        store.save("work", [])
        mock_memory.add_user("hi")
        result = handle_session_command("load work", store, mock_memory, "prev")
        assert result.session == "work"
        assert store.load("prev") == [{"role": "user", "content": "hi"}]

    def test_load_missing(self, store: SessionStore, mock_memory) -> None:
        result = handle_session_command("load ghost", store, mock_memory, None)
        assert result.session is None
        assert "not found" in result.message

    def test_load_without_name(self, store: SessionStore, mock_memory) -> None:
        result = handle_session_command("load", store, mock_memory, None)
        assert "Usage" in result.message

    def test_delete(self, store: SessionStore, mock_memory) -> None:
        store.save("work", [])
        result = handle_session_command("delete work", store, mock_memory, None)
        assert "Deleted session 'work'" in result.message
        assert not store.exists("work")

    def test_delete_missing(self, store: SessionStore, mock_memory) -> None:
        result = handle_session_command("delete ghost", store, mock_memory, None)
        assert "not found" in result.message

    def test_unknown_subcommand(self, store: SessionStore, mock_memory) -> None:
        result = handle_session_command("frobnicate", store, mock_memory, None)
        assert "Unknown" in result.message
        assert "Usage" in result.message


class TestSessionList:
    def test_empty(self, store: SessionStore) -> None:
        text = format_session_list(store)
        assert "No saved sessions" in text

    def test_lists_sessions(self, store: SessionStore) -> None:
        saved = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
        store.save("work", saved)
        text = format_session_list(store)
        assert "work" in text
        assert "2" in text


class TestRunCLI:
    async def test_resume_yes_restores_and_exit_saves(
        self, store: SessionStore, mock_memory, monkeypatch
    ) -> None:
        msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
        store.save("last", msgs)
        _script_prompt(monkeypatch, ["y", "/exit"])

        await cli_module.run_cli(_OrchStub(mock_memory), store)

        assert mock_memory.messages == msgs  # resumed
        assert store.load("last") == msgs  # checkpointed again on exit

    async def test_resume_no_starts_empty(
        self, store: SessionStore, mock_memory, monkeypatch
    ) -> None:
        store.save("last", [{"role": "user", "content": "hi"}])
        _script_prompt(monkeypatch, ["n", "/exit"])

        await cli_module.run_cli(_OrchStub(mock_memory), store)

        assert mock_memory.messages == []

    async def test_session_command_dispatches_through_loop(
        self, store: SessionStore, mock_memory, monkeypatch
    ) -> None:
        _script_prompt(monkeypatch, ["n", "/session new demo", "/exit"])

        await cli_module.run_cli(_OrchStub(mock_memory), store)

        assert store.exists("demo")

    async def test_task_create_dispatches_through_loop(
        self, store: SessionStore, mock_memory, task_store: TaskStore, monkeypatch
    ) -> None:
        _script_prompt(monkeypatch, ["n", "/task create demo task", "/exit"])

        await cli_module.run_cli(_OrchStub(mock_memory), store, task_store)

        assert len(task_store.list()) == 1
        assert task_store.list()[0].title == "demo task"

    async def test_tasks_command_dispatches_through_loop(
        self, store: SessionStore, mock_memory, task_store: TaskStore, monkeypatch
    ) -> None:
        task_store.create("existing")
        _script_prompt(monkeypatch, ["n", "/tasks", "/exit"])

        await cli_module.run_cli(_OrchStub(mock_memory), store, task_store)

        assert len(task_store.list()) == 1  # listing does not mutate

    async def test_tasks_unavailable_without_task_store(
        self, store: SessionStore, mock_memory, monkeypatch
    ) -> None:
        _script_prompt(monkeypatch, ["n", "/tasks", "/exit"])

        await cli_module.run_cli(_OrchStub(mock_memory), store)

        # No exception: the guard must short-circuit before the agent path.


class TestTaskCommandHandler:
    def test_empty_command_shows_usage(self, task_store: TaskStore) -> None:
        result = handle_task_command("", task_store)
        assert "Usage" in result.message

    def test_help_shows_usage(self, task_store: TaskStore) -> None:
        result = handle_task_command("help", task_store)
        assert "Usage" in result.message

    def test_create_multi_word_title(self, task_store: TaskStore) -> None:
        result = handle_task_command("create Buy milk", task_store)
        assert "Created task t1: Buy milk" in result.message
        assert task_store.get("t1").title == "Buy milk"

    def test_create_requires_title(self, task_store: TaskStore) -> None:
        result = handle_task_command("create", task_store)
        assert "Task title required" in result.message
        assert task_store.list() == []

    def test_create_accepts_priority_via_update_not_create(self, task_store: TaskStore) -> None:
        # create only takes a title; priority is set via update.
        result = handle_task_command("create Buy milk", task_store)
        assert "Created task t1: Buy milk (status: todo, priority: medium)" in result.message

    def test_list_empty(self, task_store: TaskStore) -> None:
        result = handle_task_command("list", task_store)
        assert "No tasks." in result.message

    def test_list_returns_tasks(self, task_store: TaskStore) -> None:
        task_store.create("First")
        task_store.create("Second")
        result = handle_task_command("list", task_store)
        assert "2 task(s):" in result.message
        assert "First" in result.message
        assert "Second" in result.message

    def test_list_filters_by_status(self, task_store: TaskStore) -> None:
        task_store.create("First")
        task_store.create("Second")
        handle_task_command("complete t1", task_store)
        result = handle_task_command("list done", task_store)
        assert "t1" in result.message
        assert "t2" not in result.message

    def test_list_invalid_status(self, task_store: TaskStore) -> None:
        result = handle_task_command("list bogus", task_store)
        assert "Invalid status" in result.message

    def test_get_returns_task(self, task_store: TaskStore) -> None:
        task_store.create("Buy milk")
        result = handle_task_command("get t1", task_store)
        assert "t1: Buy milk" in result.message

    def test_get_missing(self, task_store: TaskStore) -> None:
        result = handle_task_command("get t99", task_store)
        assert "not found" in result.message

    def test_get_without_id(self, task_store: TaskStore) -> None:
        result = handle_task_command("get", task_store)
        assert "Usage" in result.message

    def test_update_status(self, task_store: TaskStore) -> None:
        task_store.create("Buy milk")
        result = handle_task_command("update t1 status=done", task_store)
        assert "Updated task t1" in result.message
        assert task_store.get("t1").status == "done"

    def test_update_multi_word_title(self, task_store: TaskStore) -> None:
        task_store.create("Buy milk")
        result = handle_task_command("update t1 title=Bring snacks", task_store)
        assert "Updated task t1: Bring snacks" in result.message
        assert task_store.get("t1").title == "Bring snacks"

    def test_update_unknown_field(self, task_store: TaskStore) -> None:
        task_store.create("Buy milk")
        result = handle_task_command("update t1 bogus=1", task_store)
        assert "Unknown update field(s): bogus" in result.message
        assert task_store.get("t1").title == "Buy milk"

    def test_update_without_assignments(self, task_store: TaskStore) -> None:
        task_store.create("Buy milk")
        result = handle_task_command("update t1", task_store)
        assert "Usage" in result.message

    def test_update_missing(self, task_store: TaskStore) -> None:
        result = handle_task_command("update t99 status=done", task_store)
        assert "not found" in result.message

    def test_update_invalid_status(self, task_store: TaskStore) -> None:
        task_store.create("Buy milk")
        result = handle_task_command("update t1 status=bogus", task_store)
        assert "Invalid status" in result.message

    def test_complete_marks_done(self, task_store: TaskStore) -> None:
        task_store.create("Buy milk")
        result = handle_task_command("complete t1", task_store)
        assert "Completed task t1" in result.message
        assert task_store.get("t1").status == "done"

    def test_complete_missing(self, task_store: TaskStore) -> None:
        result = handle_task_command("complete t99", task_store)
        assert "not found" in result.message

    def test_delete_removes_task(self, task_store: TaskStore) -> None:
        task_store.create("Buy milk")
        result = handle_task_command("delete t1", task_store)
        assert "Deleted task t1" in result.message
        assert task_store.list() == []

    def test_delete_missing(self, task_store: TaskStore) -> None:
        result = handle_task_command("delete t99", task_store)
        assert "not found" in result.message

    def test_unknown_subcommand(self, task_store: TaskStore) -> None:
        result = handle_task_command("frobnicate", task_store)
        assert "Unknown /task command 'frobnicate'" in result.message
        assert "Usage" in result.message


class TestTaskFormatting:
    def test_format_task_list_empty(self, task_store: TaskStore) -> None:
        assert format_task_list(task_store.list()) == "No tasks."

    def test_format_task_list_contains_ids_and_status(self, task_store: TaskStore) -> None:
        task_store.create("First")
        task_store.create("Second")
        handle_task_command("complete t1", task_store)
        text = format_task_list(task_store.list())
        assert "t1" in text
        assert "t2" in text
        assert "[done" in text

    def test_format_task_shows_fields(self, task_store: TaskStore) -> None:
        task = task_store.create("Buy milk", description="2L")
        text = format_task(task)
        assert "t1: Buy milk" in text
        assert "status: todo" in text
        assert "priority: medium" in text
        assert "created:" in text
        assert "2L" in text
