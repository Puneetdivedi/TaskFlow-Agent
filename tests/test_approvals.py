"""Tests for the interactive tool-approval policy and approver."""

from __future__ import annotations

import io
from typing import Any

import pytest
from rich.console import Console

from src.ui import approvals as approvals_module
from src.ui.approvals import (
    ALWAYS_APPROVE,
    approval_reason,
    make_approver,
)


class TestApprovalReason:
    def test_always_approve_tools_flagged(self) -> None:
        assert ALWAYS_APPROVE == frozenset({"delete_file", "move_file", "run_shell", "subagent"})
        for name in ALWAYS_APPROVE:
            assert approval_reason(name, {}) is not None

    def test_safe_tools_allowed(self) -> None:
        for name in (
            "read_file",
            "list_files",
            "search_files",
            "web_search",
            "web_fetch",
            "remember",
            "recall",
            "task",
            "yaml_read",
        ):
            assert approval_reason(name, {}) is None

    def test_write_to_new_path_allowed(self, tmp_path) -> None:
        assert (
            approval_reason("write_file", {"path": str(tmp_path / "new.txt"), "content": "x"})
            is None
        )

    def test_write_to_existing_path_flagged(self, tmp_path) -> None:
        target = tmp_path / "existing.txt"
        target.write_text("data")
        reason = approval_reason("write_file", {"path": str(target), "content": "x"})
        assert reason is not None
        assert "overwrite" in reason

    def test_yaml_write_to_existing_path_flagged(self, tmp_path) -> None:
        target = tmp_path / "existing.yaml"
        target.write_text("a: 1")
        assert approval_reason("yaml_write", {"path": str(target)}) is not None

    def test_json_write_to_existing_path_flagged(self, tmp_path) -> None:
        target = tmp_path / "existing.json"
        target.write_text("{}")
        assert approval_reason("json_write", {"path": str(target)}) is not None

    def test_json_write_to_new_path_allowed(self, tmp_path) -> None:
        assert approval_reason("json_write", {"path": str(tmp_path / "new.json")}) is None

    def test_overwrite_check_ignores_non_string_path(self) -> None:
        assert approval_reason("write_file", {"path": 123, "content": "x"}) is None

    def test_overwrite_check_ignores_absent_path(self) -> None:
        assert approval_reason("write_file", {}) is None


class TestMakeApprover:
    def test_disabled_returns_none(self) -> None:
        assert make_approver(Console(file=io.StringIO()), enabled=False) is None

    def test_defaults_to_enabled(self) -> None:
        assert make_approver(Console(file=io.StringIO())) is not None

    async def test_safe_call_does_not_prompt(self, monkeypatch) -> None:
        def _boom(*args: Any, **kwargs: Any) -> str:
            raise AssertionError("safe calls must not prompt")

        monkeypatch.setattr(approvals_module.Prompt, "ask", _boom)
        approver = make_approver(Console(file=io.StringIO()))
        assert approver is not None
        assert await approver("read_file", {}) is None

    @pytest.mark.parametrize("answer", ["y", "yes", "Y"])
    async def test_yes_allows(self, monkeypatch, answer: str) -> None:
        monkeypatch.setattr(approvals_module.Prompt, "ask", lambda *a, **k: answer)
        approver = make_approver(Console(file=io.StringIO()))
        assert approver is not None
        assert await approver("delete_file", {"path": "x"}) is None

    @pytest.mark.parametrize("answer", ["n", "no", "x", ""])
    async def test_deny_returns_reason(self, monkeypatch, answer: str) -> None:
        monkeypatch.setattr(approvals_module.Prompt, "ask", lambda *a, **k: answer)
        approver = make_approver(Console(file=io.StringIO()))
        assert approver is not None
        reason = await approver("delete_file", {"path": "x"})
        assert reason is not None
        assert "delete_file" in reason

    async def test_auto_allows_rest_of_turn_without_prompting(self, monkeypatch) -> None:
        calls: list[str] = []

        def _answer(*args: Any, **kwargs: Any) -> str:
            calls.append("prompt")
            return "a"

        monkeypatch.setattr(approvals_module.Prompt, "ask", _answer)
        approver = make_approver(Console(file=io.StringIO()))
        assert approver is not None
        # First risky call prompts and is auto-allowed.
        assert await approver("delete_file", {"path": "x"}) is None
        # A second risky call is allowed without another prompt.
        assert await approver("run_shell", {"cmd": "rm -rf /"}) is None
        assert calls == ["prompt"]

    async def test_eof_denies_fail_safe(self, monkeypatch) -> None:
        def _eof(*args: Any, **kwargs: Any) -> str:
            raise EOFError

        monkeypatch.setattr(approvals_module.Prompt, "ask", _eof)
        approver = make_approver(Console(file=io.StringIO()))
        assert approver is not None
        assert await approver("delete_file", {"path": "x"}) is not None

    async def test_renders_reason_and_args(self, monkeypatch) -> None:
        monkeypatch.setattr(approvals_module.Prompt, "ask", lambda *a, **k: "n")
        buffer = io.StringIO()
        approver = make_approver(Console(file=buffer))
        assert approver is not None
        await approver("delete_file", {"path": "x"})
        rendered = buffer.getvalue()
        assert "delete_file" in rendered
        assert 'args: {"path": "x"}' in rendered
