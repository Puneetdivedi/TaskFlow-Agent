"""Tests for the AgentOrchestrator tool-calling loop (run method)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.agent.orchestrator import AgentOrchestrator
from src.tools.base import ToolError


class _ScriptedLLM:
    """LLMClient stub that returns a scripted sequence of responses."""

    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self.sent_message_lists: list[list[dict]] = []
        self.received_tool_defs: list[list[dict] | None] = []
        self.received_systems: list[str | list[dict]] = []

    async def send_messages(
        self,
        messages: list[dict] | None = None,
        system: str | None = None,
        tools: list[dict] | None = None,
    ):
        self.sent_message_lists.append(list(messages or []))
        self.received_tool_defs.append(tools)
        self.received_systems.append(system)  # type: ignore[arg-type]
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


class _StreamingLLM:
    """LLMClient stub implementing both send_messages and stream_messages."""

    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self.send_calls = 0
        self.stream_calls = 0

    async def send_messages(
        self,
        messages: list[dict] | None = None,
        system: str | None = None,
        tools: list[dict] | None = None,
    ):
        self.send_calls += 1
        if not self._responses:
            return _end_turn_response("")
        return self._responses.pop(0)

    async def stream_messages(
        self,
        messages: list[dict] | None = None,
        system: str | None = None,
        tools: list[dict] | None = None,
        *,
        on_text_delta=None,
    ):
        self.stream_calls += 1
        resp = self._responses.pop(0) if self._responses else _end_turn_response("")
        for block in resp.content:
            if block.type == "text" and on_text_delta is not None:
                await on_text_delta(block.text)
        return resp


class _InterruptingLLM:
    """LLMClient stub that interrupts mid-stream."""

    async def stream_messages(
        self,
        messages: list[dict] | None = None,
        system: str | None = None,
        tools: list[dict] | None = None,
        *,
        on_text_delta=None,
    ):
        raise KeyboardInterrupt


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


class _ScriptedSummarizer:
    """ConversationSummarizer stand-in — returns canned text or raises."""

    def __init__(self, text: str = "ROLLED UP", *, raises: bool = False) -> None:
        self.text = text
        self.raises = raises
        self.calls: list[tuple[str, str | None]] = []

    async def summarize(self, turn_text: str, existing: str | None = None) -> str:
        self.calls.append((turn_text, existing))
        if self.raises:
            raise RuntimeError("summarizer down")
        return self.text


def _long_conversation(exchanges: int = 30) -> list[dict]:
    """A long plain conversation (~3k estimated tokens) for compaction tests."""
    messages: list[dict] = []
    for i in range(exchanges):
        messages.append({"role": "user", "content": f"u{i}" + "x" * 200})
        messages.append({"role": "assistant", "content": f"a{i}" + "y" * 200})
    return messages


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


class TestRunStreaming:
    async def test_streams_deltas_forwarded_to_sink(self, mock_memory, mock_tool_registry) -> None:
        llm = _StreamingLLM([_end_turn_response("Hello there")])
        orch = _make_orchestrator(llm, mock_memory, mock_tool_registry)
        deltas: list[str] = []

        async def sink(delta: str) -> None:
            deltas.append(delta)

        result = await orch.run("hi", on_text_delta=sink)

        assert result == "Hello there"
        assert deltas == ["Hello there"]
        assert llm.stream_calls == 1
        assert llm.send_calls == 0

    async def test_on_tool_call_invoked_before_dispatch(
        self, mock_memory, mock_tool_registry
    ) -> None:
        llm = _StreamingLLM(
            [
                _tool_use_response("read_file", {"path": "a.txt"}, id_="toolu_1"),
                _end_turn_response("done"),
            ]
        )
        orch = _make_orchestrator(llm, mock_memory, mock_tool_registry)
        calls: list[tuple[str, dict]] = []

        async def sink(name: str, args: dict) -> None:
            calls.append((name, args))

        result = await orch.run("read", on_tool_call=sink)

        assert result == "done"
        assert calls == [("read_file", {"path": "a.txt"})]
        # the tool still dispatched after the callback
        assert mock_tool_registry._call_count == 1

    async def test_without_sink_uses_send_messages(self, mock_memory, mock_tool_registry) -> None:
        llm = _StreamingLLM([_end_turn_response("plain")])
        orch = _make_orchestrator(llm, mock_memory, mock_tool_registry)

        result = await orch.run("hi")

        assert result == "plain"
        assert llm.send_calls == 1
        assert llm.stream_calls == 0

    async def test_keyboard_interrupt_restores_memory(
        self, mock_memory, mock_tool_registry
    ) -> None:
        orch = _make_orchestrator(_InterruptingLLM(), mock_memory, mock_tool_registry)
        before = list(mock_memory.messages)

        async def _noop(_delta: str) -> None:
            pass

        with pytest.raises(KeyboardInterrupt):
            await orch.run("hi", on_text_delta=_noop)

        # the user message added this turn was rolled back
        assert mock_memory.messages == before


class TestSemanticMemory:
    async def test_consolidates_long_conversation(self, mock_memory, mock_tool_registry) -> None:
        long_conv = _long_conversation()
        mock_memory.restore(long_conv)
        summarizer = _ScriptedSummarizer(text="ROLLED UP")
        llm = _ScriptedLLM([_end_turn_response("done")])
        orch = AgentOrchestrator(
            llm_client=llm,
            tools=mock_tool_registry,
            memory=mock_memory,
            summarizer=summarizer,
            summary_threshold_tokens=1000,
        )

        result = await orch.run("continue")

        assert result == "done"
        assert mock_memory.summary == "ROLLED UP"
        # old turns were compacted away, but recent turns stay inline
        assert len(mock_memory.messages) < len(long_conv) + 1
        # the summary block was injected as the first system block
        injected = llm.received_systems[0]
        assert isinstance(injected, list)
        assert "[Summary of earlier conversation]" in injected[0]["text"]
        assert "ROLLED UP" in injected[0]["text"]
        assert injected[1]["text"]  # base prompt present

    async def test_no_summarizer_uses_plain_system(self, mock_memory, mock_tool_registry) -> None:
        llm = _ScriptedLLM([_end_turn_response("ok")])
        orch = _make_orchestrator(llm, mock_memory, mock_tool_registry)

        result = await orch.run("hi")

        assert result == "ok"
        assert mock_memory.summary == ""
        assert isinstance(llm.received_systems[0], str)

    async def test_threshold_zero_disables_consolidation(
        self, mock_memory, mock_tool_registry
    ) -> None:
        long_conv = _long_conversation()
        mock_memory.restore(long_conv)
        summarizer = _ScriptedSummarizer()
        llm = _ScriptedLLM([_end_turn_response("ok")])
        orch = AgentOrchestrator(
            llm_client=llm,
            tools=mock_tool_registry,
            memory=mock_memory,
            summarizer=summarizer,
            summary_threshold_tokens=0,
        )

        result = await orch.run("hi")

        assert result == "ok"
        assert summarizer.calls == []
        assert mock_memory.summary == ""
        assert len(mock_memory.messages) == len(long_conv) + 2  # user + assistant

    async def test_summarizer_failure_keeps_history(self, mock_memory, mock_tool_registry) -> None:
        long_conv = _long_conversation()
        mock_memory.restore(long_conv)
        summarizer = _ScriptedSummarizer(raises=True)
        llm = _ScriptedLLM([_end_turn_response("done")])
        orch = AgentOrchestrator(
            llm_client=llm,
            tools=mock_tool_registry,
            memory=mock_memory,
            summarizer=summarizer,
            summary_threshold_tokens=1000,
        )

        result = await orch.run("continue")

        # a failed summary never breaks the turn or mutates memory
        assert result == "done"
        assert mock_memory.summary == ""
        assert len(mock_memory.messages) == len(long_conv) + 2  # user + assistant
