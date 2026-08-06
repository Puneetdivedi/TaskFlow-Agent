"""Tests for the CLI helper functions (non-interactive parts)."""

from __future__ import annotations

import io
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from rich.console import Console

from src.memory.session_store import SessionStore
from src.memory.task_store import TaskStore
from src.ui import cli as cli_module
from src.ui.cli import (
    format_reminder_banner,
    format_session_list,
    format_task,
    format_task_list,
    handle_session_command,
    handle_task_command,
    newly_due_tasks,
    print_banner,
    print_help,
    print_tools,
)


@pytest.fixture
def store(tmp_path: Path) -> SessionStore:
    return SessionStore(db_path=tmp_path / "sessions.db")


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

    def test_save_persists_summary(self, store: SessionStore, mock_memory) -> None:
        mock_memory.add_user("hello")
        mock_memory.set_summary("ROLLED UP")
        result = handle_session_command("save work", store, mock_memory, None)
        assert "Saved session 'work'" in result.message
        assert store.load_summary("work") == "ROLLED UP"

    def test_load_restores_summary(self, store: SessionStore, mock_memory) -> None:
        store.save("work", [{"role": "user", "content": "hi"}], summary="ROLLED UP")
        mock_memory.set_summary("")
        result = handle_session_command("load work", store, mock_memory, None)
        assert result.session == "work"
        assert mock_memory.summary == "ROLLED UP"

    def test_new_checkpoint_preserves_summary(self, store: SessionStore, mock_memory) -> None:
        mock_memory.add_user("hello")
        mock_memory.set_summary("OLD")
        result = handle_session_command("new next", store, mock_memory, "prev")
        assert result.session == "next"
        assert store.load_summary("prev") == "OLD"
        # the fresh session starts summary-free
        assert mock_memory.summary == ""

    def test_load_checkpoint_preserves_summary(self, store: SessionStore, mock_memory) -> None:
        store.save("work", [])
        mock_memory.add_user("hi")
        mock_memory.set_summary("CUR")
        result = handle_session_command("load work", store, mock_memory, "prev")
        assert result.session == "work"
        assert store.load_summary("prev") == "CUR"


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

    async def test_reminders_command_dispatches_through_loop(
        self, store: SessionStore, mock_memory, task_store: TaskStore, monkeypatch
    ) -> None:
        due_date = (datetime.now() - timedelta(days=1)).date().isoformat()
        task_store.create("Overdue", due_at=due_date)
        _script_prompt(monkeypatch, ["n", "/reminders", "/exit"])

        await cli_module.run_cli(_OrchStub(mock_memory), store, task_store)

        assert task_store.get("t1").status == "todo"  # listing does not mutate

    async def test_reminders_unavailable_without_task_store(
        self, store: SessionStore, mock_memory, monkeypatch
    ) -> None:
        _script_prompt(monkeypatch, ["n", "/reminders", "/exit"])

        await cli_module.run_cli(_OrchStub(mock_memory), store)

        # No exception: the guard must short-circuit before the agent path.

    async def test_task_scheduling_through_loop(
        self, store: SessionStore, mock_memory, task_store: TaskStore, monkeypatch
    ) -> None:
        _script_prompt(
            monkeypatch,
            [
                "n",
                "/task create Foo",
                "/task update t1 due_at=2026-08-10 every_days=7",
                "/exit",
            ],
        )

        await cli_module.run_cli(_OrchStub(mock_memory), store, task_store)

        task = task_store.get("t1")
        assert task.title == "Foo"
        assert task.due_at == "2026-08-10"
        assert task.every_days == 7

    async def test_complete_recurring_through_loop(
        self, store: SessionStore, mock_memory, task_store: TaskStore, monkeypatch
    ) -> None:
        _script_prompt(
            monkeypatch,
            [
                "n",
                "/task create Foo",
                "/task update t1 due_at=2026-08-10 every_days=7",
                "/task complete t1",
                "/exit",
            ],
        )

        await cli_module.run_cli(_OrchStub(mock_memory), store, task_store)

        task = task_store.get("t1")
        assert task.status == "todo"  # rolled, not done
        assert task.due_at == "2026-08-17"


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

    def test_complete_recurring_reports_next_due(self, task_store: TaskStore) -> None:
        task_store.create("Water plants", due_at="2026-08-10", every_days=7)
        result = handle_task_command("complete t1", task_store)
        assert "(next due: 2026-08-17)" in result.message
        assert task_store.get("t1").status == "todo"

    def test_update_due_at_field(self, task_store: TaskStore) -> None:
        task_store.create("Buy milk")
        result = handle_task_command("update t1 due_at=2026-08-10", task_store)
        assert "Updated task t1" in result.message
        assert task_store.get("t1").due_at == "2026-08-10"

    def test_update_every_days_field(self, task_store: TaskStore) -> None:
        task_store.create("Buy milk")
        result = handle_task_command("update t1 every_days=7", task_store)
        assert "Updated task t1" in result.message
        assert task_store.get("t1").every_days == 7

    def test_update_invalid_every_days(self, task_store: TaskStore) -> None:
        task_store.create("Buy milk")
        result = handle_task_command("update t1 every_days=abc", task_store)
        assert "Invalid every_days" in result.message
        assert task_store.get("t1").every_days == 0

    def test_update_negative_every_days(self, task_store: TaskStore) -> None:
        task_store.create("Buy milk")
        result = handle_task_command("update t1 every_days=-1", task_store)
        assert "Invalid every_days" in result.message
        assert task_store.get("t1").every_days == 0

    def test_update_invalid_due_at(self, task_store: TaskStore) -> None:
        task_store.create("Buy milk")
        result = handle_task_command("update t1 due_at=bogus", task_store)
        assert "Invalid due_at" in result.message
        assert task_store.get("t1").due_at == ""

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

    def test_format_task_shows_due_and_recurrence(self, task_store: TaskStore) -> None:
        task = task_store.create("Water plants", due_at="2026-08-10", every_days=7)
        text = format_task(task)
        assert "due: 2026-08-10" in text
        assert "repeats every 7 day(s)" in text

    def test_format_task_list_shows_due(self, task_store: TaskStore) -> None:
        task_store.create("Water plants", due_at="2026-08-10")
        text = format_task_list(task_store.list())
        assert "due: 2026-08-10" in text


class TestReminderHelpers:
    def test_format_reminder_banner_text(self, task_store: TaskStore) -> None:
        task_store.create("Overdue")
        text = format_reminder_banner(task_store.list())
        assert "1 task(s) due or overdue" in text
        assert "t1" in text

    def test_newly_due_reports_once(self, task_store: TaskStore) -> None:
        due_date = (datetime.now() - timedelta(days=1)).date().isoformat()
        task_store.create("Overdue", due_at=due_date)
        reported: set[tuple[str, str]] = set()
        fresh = newly_due_tasks(task_store, reported)
        assert [t.id for t in fresh] == ["t1"]
        assert len(newly_due_tasks(task_store, reported)) == 0  # reported twice

    def test_newly_due_rerolled_task_reports_again(self, task_store: TaskStore) -> None:
        # A daily task fallen far behind: completing it rolls the due date by
        # one day but it is still overdue, so the new (id, due_at) pair —
        # different from the reported key — surfaces again.
        due_date = (datetime.now() - timedelta(days=10)).date().isoformat()
        task_store.create("Water plants", due_at=due_date, every_days=1)
        reported: set[tuple[str, str]] = set()
        assert len(newly_due_tasks(task_store, reported)) == 1
        task_store.complete("t1")
        fresh = newly_due_tasks(task_store, reported)
        assert len(fresh) == 1
        assert fresh[0].due_at != due_date  # new cycle, new key

    def test_newly_due_none_store(self) -> None:
        assert newly_due_tasks(None, set()) == []

    def test_startup_banner_seeds_reported(self, task_store: TaskStore) -> None:
        due_date = (datetime.now() - timedelta(days=1)).date().isoformat()
        task_store.create("Overdue", due_at=due_date)
        reported: set[tuple[str, str]] = set()
        startup_due = task_store.due()
        assert startup_due  # something is due
        reported.update((t.id, t.due_at) for t in startup_due)
        assert newly_due_tasks(task_store, reported) == []


class _StubStreamOrch:
    """Orchestrator stand-in that drives its streaming callbacks."""

    async def run(self, user_input, *, on_text_delta=None, on_tool_call=None):
        assert user_input == "hello"
        await on_text_delta("Hel")
        await on_text_delta("lo")
        await on_tool_call("read_file", {"path": "a.txt"})
        await on_text_delta(" there")
        return "Hello there"


class TestStreamingHelpers:
    def test_stream_tool_label_compact(self) -> None:
        label = cli_module._stream_tool_label("run_shell", {"cmd": "ls -la"})
        assert label == "run_shell(cmd=ls -la)"

    def test_stream_tool_label_truncates_long_values(self) -> None:
        long = "x" * 200
        label = cli_module._stream_tool_label("write_file", {"path": "a.txt", "content": long})
        assert label.startswith("write_file(path=")
        assert "…" in label
        assert len(label) < len(long)  # value was truncated, not inlined whole

    def test_should_print_final_when_streamed_as_tail(self) -> None:
        # The answer was already streamed as the tail of the panel.
        assert cli_module._should_print_final("Hi\nThere", "There") is False
        assert cli_module._should_print_final("Hello there", "Hello there") is False

    def test_should_print_final_when_not_streamed(self) -> None:
        # Error returns / stop markers are never streamed — must be printed.
        assert cli_module._should_print_final("", "Error: boom") is True
        assert cli_module._should_print_final("partial", "[Agent stopped]") is True

    def test_should_print_final_empty_response(self) -> None:
        assert cli_module._should_print_final("", "") is False

    async def test_run_streamed_turn_renders_and_returns(self) -> None:
        buffer = io.StringIO()
        console = Console(file=buffer, width=100)

        final, streamed = await cli_module._run_streamed_turn(_StubStreamOrch(), "hello", console)

        assert final == "Hello there"
        assert streamed == "Hello there"
        rendered = buffer.getvalue()
        # both the streamed text and the tool-call line reached the panel
        assert "Hello there" in rendered
        assert "read_file" in rendered
