"""Tests for the CLI helper functions (non-interactive parts)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.memory.session_store import SessionStore
from src.ui import cli as cli_module
from src.ui.cli import (
    format_session_list,
    handle_session_command,
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
