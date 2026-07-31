"""Tests for the agent orchestrator (mocked client where possible)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.agent.orchestrator import DEFAULT_SYSTEM_PROMPT, AgentOrchestrator
from src.tools.base import ToolError
from src.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Helpers — simple in-line doubles so we don't need conftest.py yet
# ---------------------------------------------------------------------------
class _MockLLMClient:
    """Minimal LLMClient stub for orchestrator tests."""

    def __init__(self) -> None:
        self._response = _make_end_turn_response("")

    async def send_messages(self, messages=None, system=None, tools=None):
        return self._response


class _MockMemory:
    """Minimal IMemory stub."""

    def __init__(self) -> None:
        self.messages: list[dict] = []
        self.history: list[tuple[str, str | list[dict]]] = []

    def add_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str | list[dict]) -> None:
        self.messages.append({"role": "assistant", "content": content})
        self.history.append(("assistant", content))

    def add_tool_result(self, tool_use_id: str, content: str) -> None:
        self.messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": content,
                    }
                ],
            }
        )

    def prune(self) -> None:
        pass

    def clear(self) -> None:
        self.messages.clear()
        self.history.clear()


class _MockToolRegistry:
    """Minimal IToolRegistry stub."""

    def __init__(self) -> None:
        self._call_count = 0

    @property
    def tool_names(self) -> list[str]:
        return ["mock_tool"]

    def anthropic_tool_defs(self) -> list[dict]:
        return [
            {
                "name": "mock_tool",
                "description": "A mock tool",
                "input_schema": {"type": "object", "properties": {}},
            }
        ]

    async def dispatch(self, name: str, arguments: dict) -> str:
        self._call_count += 1
        return "mock result"


def _make_end_turn_response(text: str):
    """Return an object duck-typing an Anthropic `Message` with end_turn."""
    content = [] if not text else [SimpleNamespace(type="text", text=text)]
    return SimpleNamespace(content=content, stop_reason="end_turn")


def _make_tool_use_response(tool_name: str, tool_input: dict, tool_id: str = "toolu_mock"):
    """Return an object duck-typing an Anthropic `Message` with a tool_use block."""
    block = SimpleNamespace(type="tool_use", id=tool_id, name=tool_name, input=tool_input)
    return SimpleNamespace(content=[block], stop_reason="tool_use")


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------
class TestToolRegistry:
    def test_has_phase1_tools(self) -> None:
        registry = ToolRegistry()
        names = registry.tool_names
        assert "read_file" in names
        assert "write_file" in names
        assert "list_files" in names
        assert "search_files" in names
        assert "run_shell" in names
        assert "move_file" in names
        assert "delete_file" in names
        assert "file_index" in names

    def test_registry_safety_level_passed(self) -> None:
        registry = ToolRegistry(safety_level=0)
        with pytest.raises(ToolError, match="blocked at safety level 0"):
            import asyncio

            asyncio.run(registry.dispatch("run_shell", {"command": "echo hello"}))

    def test_registry_default_safety_allows_shell(self) -> None:
        """Safety level 1 (default) should allow shell commands."""
        import asyncio

        result = asyncio.run(ToolRegistry().dispatch("run_shell", {"command": "echo hello"}))
        assert "hello" in result

    def test_anthropic_defs_are_valid(self) -> None:
        registry = ToolRegistry()
        defs = registry.anthropic_tool_defs()
        for d in defs:
            assert "name" in d
            assert "description" in d
            assert "input_schema" in d
            assert "type" in d["input_schema"]
            assert "properties" in d["input_schema"]

    async def test_dispatch_unknown_tool(self) -> None:
        registry = ToolRegistry()
        with pytest.raises(ToolError, match="Unknown tool"):
            await registry.dispatch("nonexistent", {})


# ---------------------------------------------------------------------------
# AgentOrchestrator (no real API call)
# ---------------------------------------------------------------------------
class TestAgentOrchestratorInit:
    def test_initializes_with_injected_deps(self) -> None:
        orch = AgentOrchestrator(
            llm_client=_MockLLMClient(),
            tools=_MockToolRegistry(),
            memory=_MockMemory(),
        )
        assert orch._client is not None
        assert orch._system_prompt == DEFAULT_SYSTEM_PROMPT
        assert orch._max_tool_calls == 25

    def test_initializes_with_custom_prompt(self) -> None:
        custom = "You are a custom agent."
        orch = AgentOrchestrator(
            llm_client=_MockLLMClient(),
            tools=_MockToolRegistry(),
            memory=_MockMemory(),
            system_prompt=custom,
        )
        assert orch._system_prompt == custom

    def test_tools_property(self) -> None:
        tools = _MockToolRegistry()
        orch = AgentOrchestrator(
            llm_client=_MockLLMClient(),
            tools=tools,
            memory=_MockMemory(),
        )
        assert orch.tools is tools
        assert isinstance(orch.tools, _MockToolRegistry)
