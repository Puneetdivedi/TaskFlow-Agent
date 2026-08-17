"""Tests for autonomous task execution — the runner and the tool."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agent.automation import AutomationRunner
from src.memory.task_store import TaskStore
from src.tools.automation_tool import AutomationRunTool
from src.tools.base import ToolError


# --- scripted LLM + response helpers (duck-type Anthropic Message) ---------
def _end_turn_response(text: str = ""):
    content = [] if not text else [SimpleNamespace(type="text", text=text)]
    return SimpleNamespace(content=content, stop_reason="end_turn")


def _tool_use_response(name: str, input_: dict, id_: str = "toolu_1"):
    block = SimpleNamespace(type="tool_use", id=id_, name=name, input=input_)
    return SimpleNamespace(content=[block], stop_reason="tool_use")


class _ScriptedLLM:
    """LLMClient stub returning a scripted sequence, recording calls."""

    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self.sent_messages: list[list[dict]] = []

    async def send_messages(self, messages=None, system=None, tools=None):
        self.sent_messages.append(list(messages or []))
        if not self._responses:
            return _end_turn_response("")
        return self._responses.pop(0)


class _FailingLLM:
    """LLMClient stub that always raises."""

    async def send_messages(self, messages=None, system=None, tools=None):
        raise RuntimeError("network down")


class _StubRegistry:
    """Minimal IToolRegistry that records dispatched calls."""

    tool_names = ["read_file", "write_file", "delete_file", "run_shell"]

    def __init__(self) -> None:
        self.dispatched: list[tuple[str, dict]] = []

    def anthropic_tool_defs(self) -> list[dict]:
        return [{"name": n, "description": "", "input_schema": {}} for n in self.tool_names]

    async def dispatch(self, name: str, arguments: dict) -> str:
        self.dispatched.append((name, arguments))
        return f"{name} result"


def _runner(llm, registry=None) -> AutomationRunner:
    return AutomationRunner(
        llm_client=llm,  # type: ignore[arg-type]
        registry=registry or _StubRegistry(),  # type: ignore[arg-type]
    )


# --- AutomationRunner -------------------------------------------------------
class TestAutomationRunner:
    async def test_returns_report_on_end_turn(self) -> None:
        llm = _ScriptedLLM([_end_turn_response("Done everything.")])
        runner = _runner(llm)

        report = await runner.run_plan("create a file")

        # Should return a formatted report with status, timing, plan, and result
        assert "Automation Report: Automation Run" in report
        assert "**Status:** ✅ SUCCESS" in report
        assert "**Started:**" in report
        assert "**Completed:**" in report
        assert "**Duration:**" in report
        assert "## Plan" in report
        assert "create a file" in report
        assert "## Result" in report
        assert "Done everything." in report
        # fresh, isolated context seeded with the plan as the only user message
        assert llm.sent_messages[0] == [{"role": "user", "content": "create a file"}]

    async def test_seeds_title_and_plan(self) -> None:
        llm = _ScriptedLLM([_end_turn_response("ok")])
        runner = _runner(llm)

        await runner.run_plan("do it", title="My task")

        assert llm.sent_messages[0][0]["content"] == "# My task\n\ndo it"

    async def test_dispatches_tools(self) -> None:
        registry = _StubRegistry()
        llm = _ScriptedLLM(
            [
                _tool_use_response("write_file", {"path": "a.txt", "content": "x"}),
                _end_turn_response("file written"),
            ]
        )
        runner = _runner(llm, registry)

        report = await runner.run_plan("write a file")

        assert "Automation Report: Automation Run" in report
        assert "**Status:** ✅ SUCCESS" in report
        assert "## Result" in report
        assert "file written" in report
        assert registry.dispatched == [("write_file", {"path": "a.txt", "content": "x"})]

    async def test_destructive_call_denied_by_safe_policy(self) -> None:
        llm = _ScriptedLLM(
            [
                _tool_use_response("delete_file", {"path": "x.txt"}),
                _end_turn_response("noted"),
            ]
        )
        registry = _StubRegistry()
        runner = _runner(llm, registry)

        await runner.run_plan("delete something")

        assert registry.dispatched == []  # denied before dispatch
        # The second call's last message is the user message carrying the
        # tool_result blocks for the denied call.
        last = llm.sent_messages[1][-1]
        tool_results = last["content"]
        assert "Autonomous policy" in tool_results[0]["content"]

    async def test_api_failure_returns_graceful_report(self) -> None:
        # The orchestrator catches the API error itself and returns an error
        # string — the runner never raises, the caller always gets a report.
        runner = _runner(_FailingLLM())

        report = await runner.run_plan("go")

        assert isinstance(report, str)
        assert "network down" in report


# --- AutomationRunTool ------------------------------------------------------
class _FakeRunner:
    def __init__(self, result: str = "report text") -> None:
        self.result = result
        self.calls: list[tuple[str, str]] = []

    async def run_plan(self, plan: str, title: str = "") -> str:
        self.calls.append((plan, title))
        return self.result


def _task_store(tmp_path: Path) -> TaskStore:
    return TaskStore(db_path=tmp_path / "tasks.db")


class TestAutomationRunTool:
    async def test_denies_task_without_plan(self, tmp_path: Path) -> None:
        store = _task_store(tmp_path)
        store.create("No plan")
        tool = AutomationRunTool(_FakeRunner(), store)  # type: ignore[arg-type]

        with pytest.raises(ToolError, match="no plan"):
            await tool.run("t1")

    async def test_runs_plan_and_advances_recurring(self, tmp_path: Path) -> None:
        store = _task_store(tmp_path)
        store.create(
            "Water plants",
            plan="write a file",
            auto_run=True,
            due_at="2026-08-10",
            every_days=7,
        )
        runner = _FakeRunner()
        tool = AutomationRunTool(runner, store)  # type: ignore[arg-type]

        result = await tool.run("t1")

        assert runner.calls == [("write a file", "Water plants")]
        assert "report text" in result
        task = store.get("t1")
        assert task.due_at == "2026-08-17"  # rolled forward
        assert task.status == "todo"  # recurring cycle continues

    async def test_runs_plan_and_advances_one_shot(self, tmp_path: Path) -> None:
        store = _task_store(tmp_path)
        store.create("One shot", plan="do something", due_at="2026-08-10")
        tool = AutomationRunTool(_FakeRunner(), store)  # type: ignore[arg-type]

        await tool.run("t1")

        assert store.get("t1").status == "done"

    async def test_missing_task_raises(self, tmp_path: Path) -> None:
        store = _task_store(tmp_path)
        tool = AutomationRunTool(_FakeRunner(), store)  # type: ignore[arg-type]

        with pytest.raises(ToolError, match="not found"):
            await tool.run("t99")
