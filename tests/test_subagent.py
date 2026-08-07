"""Tests for sub-agents — presets, the isolated runner loop, and the tool."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.agent.subagent import (
    SUBAGENT_TOOL_NAME,
    SubAgent,
    SubAgentRunner,
    default_subagents,
)
from src.tools.subagent_tool import SubAgentTool


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
        self.sent_systems: list[str | list[dict]] = []
        self.sent_tools: list[list[dict] | None] = []

    async def send_messages(self, messages=None, system=None, tools=None):
        self.sent_messages.append(list(messages or []))
        self.sent_systems.append(system)  # type: ignore[arg-type]
        self.sent_tools.append(tools)
        if not self._responses:
            return _end_turn_response("")
        return self._responses.pop(0)


class _FailingLLM:
    """LLMClient stub that always raises."""

    async def send_messages(self, messages=None, system=None, tools=None):
        raise RuntimeError("network down")


class _NeverCalledLLM:
    async def send_messages(self, messages=None, system=None, tools=None):
        raise AssertionError("LLM should not be called for pure planning queries")


class _StubRegistry:
    """Minimal IToolRegistry with a subagent tool in its list."""

    tool_names = ["subagent", "read_file", "write_file", "run_shell"]

    def anthropic_tool_defs(self) -> list[dict]:
        return [{"name": n, "description": "", "input_schema": {}} for n in self.tool_names]

    async def dispatch(self, name: str, arguments: dict) -> str:
        raise AssertionError(f"dispatch should not be called here ({name})")


def _runner(
    llm,
    registry=None,
    agents=None,
    *,
    max_steps: int = 12,
) -> SubAgentRunner:
    return SubAgentRunner(
        llm_client=llm,  # type: ignore[arg-type]
        registry=registry or _StubRegistry(),  # type: ignore[arg-type]
        subagents=agents or default_subagents(),
        max_steps=max_steps,
    )


# --- presets ---------------------------------------------------------------
class TestDefaultSubagents:
    def test_three_builtin_roles(self) -> None:
        agents = default_subagents()
        assert set(agents) == {"researcher", "coder", "reviewer"}

    def test_researcher_has_web_but_no_shell(self) -> None:
        tools = set(default_subagents()["researcher"].tools)
        assert {"web_search", "web_fetch", "read_file"}.issubset(tools)
        assert "run_shell" not in tools

    def test_coder_has_shell_and_file_writes(self) -> None:
        tools = set(default_subagents()["coder"].tools)
        assert {"write_file", "run_shell", "yaml_write"}.issubset(tools)
        assert "web_search" not in tools

    def test_reviewer_is_read_only_plus_shell(self) -> None:
        tools = set(default_subagents()["reviewer"].tools)
        assert {"read_file", "search_files", "run_shell"}.issubset(tools)
        assert "write_file" not in tools


# --- SubAgentRunner --------------------------------------------------------
class TestSubAgentRunner:
    async def test_returns_text_on_end_turn(self, mock_tool_registry) -> None:
        llm = _ScriptedLLM([_end_turn_response("All done.")])
        runner = _runner(llm, registry=mock_tool_registry)

        result = await runner.run("researcher", "find the answer")

        assert result == "All done."
        # the task seeded a fresh, isolated context
        assert llm.sent_messages[0][0] == {"role": "user", "content": "find the answer"}
        assert len(llm.sent_messages[0]) == 1

    async def test_dispatches_tool_then_returns(self, mock_tool_registry) -> None:
        llm = _ScriptedLLM(
            [
                _tool_use_response("mock_tool", {"a": 1}),
                _end_turn_response("report ready"),
            ]
        )
        agents = {"worker": SubAgent(name="worker", system_prompt="work", tools=("mock_tool",))}
        runner = _runner(llm, registry=mock_tool_registry, agents=agents)

        result = await runner.run("worker", "do a thing")

        assert result == "report ready"
        assert mock_tool_registry._call_count == 1
        # the tool_result was fed back to the model
        assert any(
            isinstance(m.get("content"), list)
            and any(c.get("type") == "tool_result" for c in m["content"])
            for m in llm.sent_messages[1]
        )

    async def test_rejects_disallowed_tool(self, mock_tool_registry) -> None:
        # The role only allows mock_tool; the (misbehaving) model calls run_shell.
        llm = _ScriptedLLM(
            [
                _tool_use_response("run_shell", {"cmd": "rm -rf /"}),
                _end_turn_response("noted"),
            ]
        )
        agents = {"solo": SubAgent(name="solo", system_prompt="be safe", tools=("mock_tool",))}
        runner = _runner(llm, registry=mock_tool_registry, agents=agents)

        result = await runner.run("solo", "go")

        assert result == "noted"
        assert mock_tool_registry._call_count == 0  # never dispatched
        assert "not available" in str(llm.sent_messages[1][-1])

    async def test_all_tools_default_excludes_subagent(self) -> None:
        # empty tools -> "all registered tools" is the default contract
        spec = SubAgent(name="solo", system_prompt="p")
        runner = _runner(_NeverCalledLLM(), agents={spec.name: spec})
        # empty-tools default expands to everything except the subagent tool
        allowed = runner._allowed_for(spec)
        assert SUBAGENT_TOOL_NAME not in allowed
        assert "read_file" in allowed
        assert not any(d["name"] == SUBAGENT_TOOL_NAME for d in runner._defs_for(spec))

    async def test_unknown_agent_raises_key_error(self, mock_tool_registry) -> None:
        runner = _runner(_NeverCalledLLM(), registry=mock_tool_registry)

        with pytest.raises(KeyError):
            await runner.run("ghost", "x")

    async def test_api_error_returns_text_not_raise(self, mock_tool_registry) -> None:
        runner = _runner(_FailingLLM(), registry=mock_tool_registry)

        result = await runner.run("researcher", "go")

        assert "failed" in result
        assert result.startswith("(Sub-agent")

    async def test_step_budget_truncates(self, mock_tool_registry) -> None:
        llm = _ScriptedLLM([_tool_use_response("mock_tool", {})] * 10)
        agents = {"worker": SubAgent(name="worker", system_prompt="work", tools=("mock_tool",))}
        runner = _runner(llm, registry=mock_tool_registry, agents=agents, max_steps=3)

        result = await runner.run("worker", "loop forever")

        assert "step budget" in result
        assert "truncated" in result
        assert mock_tool_registry._call_count == 3


# --- SubAgentTool ----------------------------------------------------------
class _FakeRunner:
    def __init__(self, result: str = "report") -> None:
        self.agent_names = ["researcher", "coder"]
        self.result = result
        self.calls: list[tuple[str, str]] = []

    async def run(self, name: str, task: str) -> str:
        self.calls.append((name, task))
        if name == "ghost":
            raise KeyError("ghost")
        if name == "boom":
            raise RuntimeError("kaboom")
        return self.result


class TestSubAgentTool:
    async def test_delegates_and_returns_report(self) -> None:
        runner = _FakeRunner(result="findings")
        tool = SubAgentTool(runner)  # type: ignore[arg-type]

        result = await tool.run(name="researcher", task="find X")

        assert result == "findings"
        assert runner.calls == [("researcher", "find X")]

    async def test_unknown_name_returns_error_string(self) -> None:
        tool = SubAgentTool(_FakeRunner())  # type: ignore[arg-type]

        result = await tool.run(name="ghost", task="x")

        assert result.startswith("Error: unknown sub-agent")
        assert "researcher" in result  # lists the available names

    async def test_runner_failure_returns_error_string(self) -> None:
        tool = SubAgentTool(_FakeRunner())  # type: ignore[arg-type]

        result = await tool.run(name="boom", task="x")

        assert result == "Error: sub-agent 'boom' failed: kaboom"

    def test_schema_exposes_agent_enum(self) -> None:
        tool = SubAgentTool(_FakeRunner())  # type: ignore[arg-type]
        names = tool.input_schema["properties"]["name"]["enum"]
        assert set(names) == {"researcher", "coder"}
        assert tool.input_schema["required"] == ["name", "task"]
