"""Tests for the AgentOrchestrator tool-calling loop (run method)."""

from __future__ import annotations

from types import SimpleNamespace

from src.agent.orchestrator import AgentOrchestrator
from src.tools.base import ToolError


class _ScriptedLLM:
    """LLMClient stub that returns a scripted sequence of responses."""

    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self.sent_message_lists: list[list[dict]] = []
        self.received_tool_defs: list[list[dict] | None] = []

    async def send_messages(
        self,
        messages: list[dict] | None = None,
        system: str | None = None,
        tools: list[dict] | None = None,
    ):
        self.sent_message_lists.append(list(messages or []))
        self.received_tool_defs.append(tools)
        if not self._responses:
            return _end_turn_response("")
        return self._responses.pop(0)


class _FailingToolRegistry:
    """IToolRegistry stub that raises ToolError for every dispatch."""

    @property
    def tool_names(self) -> list[str]:
        return ["broken_tool"]

    def anthropic_tool_defs(self) -> list[dict]:
        return [{"name": "broken_tool", "description": "", "input_schema": {"type": "object"}}]

    async def dispatch(self, name: str, arguments: dict) -> str:
        raise ToolError(f"{name} exploded")


class _FailingLLM:
    """LLMClient stub that always raises."""

    async def send_messages(
        self,
        messages: list[dict] | None = None,
        system: str | None = None,
        tools: list[dict] | None = None,
    ):
        raise RuntimeError("network down")


# --- response helpers (duck-type Anthropic Message) ---
def _end_turn_response(text: str = ""):
    content = [] if not text else [SimpleNamespace(type="text", text=text)]
    return SimpleNamespace(content=content, stop_reason="end_turn")


def _tool_use_response(name: str, input_: dict, id_: str = "toolu_1"):
    block = SimpleNamespace(type="tool_use", id=id_, name=name, input=input_)
    return SimpleNamespace(content=[block], stop_reason="tool_use")


def _make_orchestrator(llm, memory, tools) -> AgentOrchestrator:
    return AgentOrchestrator(
        llm_client=llm,
        tools=tools,
        memory=memory,
    )


class TestRunEndToEnd:
    async def test_returns_text_on_end_turn(self, mock_memory, mock_tool_registry) -> None:
        llm = _ScriptedLLM([_end_turn_response("Hello world")])
        orch = _make_orchestrator(llm, mock_memory, mock_tool_registry)

        result = await orch.run("hi")

        assert result == "Hello world"
        # user message was recorded
        assert mock_memory.messages[0]["role"] == "user"
        assert mock_memory.messages[0]["content"] == "hi"
        # assistant text was recorded
        assert mock_memory.messages[-1]["role"] == "assistant"

    async def test_tool_call_then_end_turn(self, mock_memory, mock_tool_registry) -> None:
        llm = _ScriptedLLM(
            [
                _tool_use_response("read_file", {"path": "a.txt"}, id_="toolu_1"),
                _end_turn_response("I read the file"),
            ]
        )
        orch = _make_orchestrator(llm, mock_memory, mock_tool_registry)

        result = await orch.run("read the file")

        assert result == "I read the file"
        assert mock_tool_registry._call_count == 1
        # tool_result was appended to memory
        tool_msgs = [
            m
            for m in mock_memory.messages
            if isinstance(m.get("content"), list) and m["content"][0].get("type") == "tool_result"
        ]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["content"][0]["tool_use_id"] == "toolu_1"
        # both LLM calls carried the tool definitions
        assert llm.received_tool_defs[0] is not None

    async def test_tool_error_is_surfaced_to_llm(self, mock_memory) -> None:
        llm = _ScriptedLLM(
            [
                _tool_use_response("broken_tool", {}, id_="toolu_9"),
                _end_turn_response("it failed"),
            ]
        )
        orch = _make_orchestrator(llm, mock_memory, _FailingToolRegistry())

        result = await orch.run("run broken tool")

        assert result == "it failed"
        tool_msgs = [
            m
            for m in mock_memory.messages
            if isinstance(m.get("content"), list) and m["content"][0].get("type") == "tool_result"
        ]
        assert "Error:" in tool_msgs[0]["content"][0]["content"]

    async def test_api_error_returns_error_text(self, mock_memory, mock_tool_registry) -> None:
        orch = _make_orchestrator(_FailingLLM(), mock_memory, mock_tool_registry)

        result = await orch.run("ping")

        assert result.startswith("API error:")
        # assistant error was recorded so the conversation stays coherent
        assert mock_memory.messages[-1]["role"] == "assistant"
        assert "network down" in mock_memory.messages[-1]["content"]

    async def test_stop_sequence_returns_marker(self, mock_memory, mock_tool_registry) -> None:
        response = SimpleNamespace(content=[], stop_reason="stop_sequence")
        llm = _ScriptedLLM([response])
        orch = _make_orchestrator(llm, mock_memory, mock_tool_registry)

        result = await orch.run("stop please")

        assert "stop sequence" in result


class TestMaxToolCalls:
    async def test_aborts_when_max_tool_calls_exceeded(
        self, mock_memory, mock_tool_registry
    ) -> None:
        # LLM keeps requesting tools forever — orchestrator must abort.
        llm = _ScriptedLLM([_tool_use_response("read_file", {"path": "x"})] * 10)
        orch = _make_orchestrator(llm, mock_memory, mock_tool_registry)
        orch._max_tool_calls = 3

        result = await orch.run("loop forever")

        assert "Exceeded max tool calls" in result
        # The aborting message was fed back to the model as a tool_result
        tool_msgs = [
            m
            for m in mock_memory.messages
            if isinstance(m.get("content"), list) and m["content"][0].get("type") == "tool_result"
        ]
        assert any("Exceeded max tool calls" in m["content"][0]["content"] for m in tool_msgs)
