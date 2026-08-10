"""Tests for the AgentOrchestrator tool-calling loop (run method)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from src.agent.orchestrator import AgentOrchestrator
from src.interfaces.usage import Usage
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


class _SelectiveFailingRegistry:
    """IToolRegistry stub where ``bad_tool`` raises a non-ToolError exception."""

    def __init__(self) -> None:
        self._ok_count = 0

    @property
    def tool_names(self) -> list[str]:
        return ["ok_tool", "bad_tool"]

    def anthropic_tool_defs(self) -> list[dict]:
        return [
            {"name": "ok_tool", "description": "", "input_schema": {"type": "object"}},
            {"name": "bad_tool", "description": "", "input_schema": {"type": "object"}},
        ]

    async def dispatch(self, name: str, arguments: dict) -> str:
        if name == "bad_tool":
            raise ValueError("bad tool exploded")
        self._ok_count += 1
        return f"ok:{name}"


class _TrackingToolRegistry:
    """IToolRegistry stub that records peak concurrent in-flight dispatches."""

    def __init__(self) -> None:
        self._call_count = 0
        self._in_flight = 0
        self._peak_in_flight = 0

    @property
    def tool_names(self) -> list[str]:
        return ["mock_tool"]

    def anthropic_tool_defs(self) -> list[dict]:
        return [{"name": "mock_tool", "description": "", "input_schema": {"type": "object"}}]

    async def dispatch(self, name: str, arguments: dict) -> str:
        self._call_count += 1
        self._in_flight += 1
        self._peak_in_flight = max(self._peak_in_flight, self._in_flight)
        await asyncio.sleep(0.01)  # yield so a concurrent batch actually overlaps
        self._in_flight -= 1
        return f"result:{name}"


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


class _BilledLLM:
    """LLMClient stub exposing ``usage`` / ``estimated_cost`` like ClaudeClient."""

    def __init__(self, responses: list, usage: Usage, cost: float) -> None:
        self._responses = list(responses)
        self.usage = usage
        self._cost = cost

    async def send_messages(
        self,
        messages: list[dict] | None = None,
        system: str | None = None,
        tools: list[dict] | None = None,
    ):
        if not self._responses:
            return _end_turn_response("")
        return self._responses.pop(0)

    def estimated_cost(self) -> float:
        return self._cost


# --- response helpers (duck-type Anthropic Message) ---
def _end_turn_response(text: str = ""):
    content = [] if not text else [SimpleNamespace(type="text", text=text)]
    return SimpleNamespace(content=content, stop_reason="end_turn")


def _tool_use_response(name: str, input_: dict, id_: str = "toolu_1"):
    block = SimpleNamespace(type="tool_use", id=id_, name=name, input=input_)
    return SimpleNamespace(content=[block], stop_reason="tool_use")


def _multi_tool_use_response(blocks: list[tuple[str, str, dict]]):
    """A Message with several ``(id, name, input)`` tool_use blocks."""
    content = [
        SimpleNamespace(type="tool_use", id=id_, name=name, input=input_)
        for id_, name, input_ in blocks
    ]
    return SimpleNamespace(content=content, stop_reason="tool_use")


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


class TestParallelToolCalls:
    async def test_all_blocks_dispatch_in_one_ordered_user_message(
        self, mock_memory, mock_tool_registry
    ) -> None:
        llm = _ScriptedLLM(
            [
                _multi_tool_use_response(
                    [
                        ("toolu_a", "read_file", {"path": "a.txt"}),
                        ("toolu_b", "read_file", {"path": "b.txt"}),
                        ("toolu_c", "read_file", {"path": "c.txt"}),
                    ]
                ),
                _end_turn_response("done"),
            ]
        )
        orch = _make_orchestrator(llm, mock_memory, mock_tool_registry)

        result = await orch.run("read all")

        assert result == "done"
        assert mock_tool_registry._call_count == 3
        tool_msgs = [
            m
            for m in mock_memory.messages
            if isinstance(m.get("content"), list) and m["content"][0].get("type") == "tool_result"
        ]
        # All three results land in a SINGLE user message, in block order.
        assert len(tool_msgs) == 1
        blocks = tool_msgs[0]["content"]
        assert [b["tool_use_id"] for b in blocks] == ["toolu_a", "toolu_b", "toolu_c"]
        assert [b["content"] for b in blocks] == ["mock result"] * 3

    async def test_on_tool_call_fires_for_each_block_in_order(
        self, mock_memory, mock_tool_registry
    ) -> None:
        llm = _ScriptedLLM(
            [
                _multi_tool_use_response(
                    [
                        ("toolu_a", "read_file", {"path": "a.txt"}),
                        ("toolu_b", "read_file", {"path": "b.txt"}),
                    ]
                ),
                _end_turn_response("done"),
            ]
        )
        orch = _make_orchestrator(llm, mock_memory, mock_tool_registry)
        calls: list[tuple[str, dict]] = []

        async def sink(name: str, args: dict) -> None:
            calls.append((name, args))

        result = await orch.run("read", on_tool_call=sink)

        assert result == "done"
        assert calls == [
            ("read_file", {"path": "a.txt"}),
            ("read_file", {"path": "b.txt"}),
        ]

    async def test_failing_tool_becomes_error_without_cancelling_siblings(
        self, mock_memory
    ) -> None:
        llm = _ScriptedLLM(
            [
                _multi_tool_use_response(
                    [
                        ("toolu_ok1", "ok_tool", {}),
                        ("toolu_bad", "bad_tool", {}),
                        ("toolu_ok2", "ok_tool", {}),
                    ]
                ),
                _end_turn_response("done"),
            ]
        )
        registry = _SelectiveFailingRegistry()
        orch = _make_orchestrator(llm, mock_memory, registry)

        result = await orch.run("mixed")

        assert result == "done"
        tool_msgs = [
            m
            for m in mock_memory.messages
            if isinstance(m.get("content"), list) and m["content"][0].get("type") == "tool_result"
        ]
        by_id = {b["tool_use_id"]: b["content"] for b in tool_msgs[0]["content"]}
        assert by_id["toolu_ok1"] == "ok:ok_tool"
        assert "Error: bad tool exploded" in by_id["toolu_bad"]
        assert by_id["toolu_ok2"] == "ok:ok_tool"
        # the failing tool didn't cancel its siblings or crash the turn
        assert registry._ok_count == 2

    async def test_parallel_cap_limits_concurrent_dispatch(self, mock_memory) -> None:
        llm = _ScriptedLLM(
            [
                _multi_tool_use_response(
                    [
                        ("toolu_1", "mock_tool", {}),
                        ("toolu_2", "mock_tool", {}),
                        ("toolu_3", "mock_tool", {}),
                        ("toolu_4", "mock_tool", {}),
                    ]
                ),
                _end_turn_response("done"),
            ]
        )
        registry = _TrackingToolRegistry()
        orch = _make_orchestrator(llm, mock_memory, registry)
        orch._max_parallel_tool_calls = 2

        result = await orch.run("parallel")

        assert result == "done"
        # every block still dispatched, in two chunks of two
        assert registry._call_count == 4
        assert registry._peak_in_flight == 2

    async def test_single_block_response_unchanged(self, mock_memory, mock_tool_registry) -> None:
        """One tool_use block per response behaves exactly as before."""
        llm = _ScriptedLLM(
            [
                _tool_use_response("read_file", {"path": "a.txt"}, id_="toolu_1"),
                _end_turn_response("I read the file"),
            ]
        )
        orch = _make_orchestrator(llm, mock_memory, mock_tool_registry)

        result = await orch.run("read")

        assert result == "I read the file"
        assert mock_tool_registry._call_count == 1
        tool_msgs = [
            m
            for m in mock_memory.messages
            if isinstance(m.get("content"), list) and m["content"][0].get("type") == "tool_result"
        ]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["content"][0]["tool_use_id"] == "toolu_1"


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


class TestUsageAndBudget:
    """Orchestrator usage accessors and the MAX_COST_USD budget halt."""

    async def test_usage_and_cost_mirror_client(self, mock_memory, mock_tool_registry) -> None:
        usage = Usage(input_tokens=100, output_tokens=20)
        llm = _BilledLLM([_end_turn_response("hi")], usage=usage, cost=0.5)
        orch = _make_orchestrator(llm, mock_memory, mock_tool_registry)

        result = await orch.run("hi")

        assert result == "hi"
        assert orch.usage == usage
        assert orch.estimated_cost() == 0.5

    async def test_client_without_usage_degrades_to_empty(
        self, mock_memory, mock_tool_registry
    ) -> None:
        # _ScriptedLLM has no .usage / .estimated_cost — must not crash.
        llm = _ScriptedLLM([_end_turn_response("ok")])
        orch = _make_orchestrator(llm, mock_memory, mock_tool_registry)

        result = await orch.run("hi")

        assert result == "ok"
        assert orch.usage == Usage()
        assert orch.estimated_cost() == 0.0

    async def test_budget_exhausted_stops_before_tools(
        self, mock_memory, mock_tool_registry
    ) -> None:
        # A tool request sits first in the script — if tools ran, _call_count
        # would be 1. The budget halt must return before any dispatch.
        llm = _BilledLLM(
            [_tool_use_response("read_file", {"path": "x"}, id_="toolu_1")],
            usage=Usage(input_tokens=1_000_000),
            cost=3.0,
        )
        orch = AgentOrchestrator(
            llm_client=llm,
            tools=mock_tool_registry,
            memory=mock_memory,
            cost_budget_usd=1.0,
        )

        result = await orch.run("do it")

        assert "Budget exhausted" in result
        assert "$1.00 cap" in result
        assert mock_tool_registry._call_count == 0
        # the halt message was recorded as the assistant turn
        assert mock_memory.messages[-1]["role"] == "assistant"

    async def test_budget_zero_disables_cap(self, mock_memory, mock_tool_registry) -> None:
        llm = _BilledLLM(
            [_end_turn_response("done")],
            usage=Usage(input_tokens=1_000_000),
            cost=3.0,
        )
        orch = AgentOrchestrator(
            llm_client=llm,
            tools=mock_tool_registry,
            memory=mock_memory,
            cost_budget_usd=0.0,
        )

        result = await orch.run("do it")

        assert result == "done"
