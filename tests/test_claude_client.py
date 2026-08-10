"""Tests for the ClaudeClient API wrapper (retries and error handling)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from anthropic import (
    APIError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)

from src.agent.claude_client import ClaudeClient, ClaudeClientError
from src.interfaces.usage import Usage


def _message_response(usage: Any = None) -> SimpleNamespace:
    """A minimal duck-typed Anthropic ``Message``.

    *usage* defaults to ``None`` so existing tests keep exercising the
    no-usage path; usage-accumulation tests pass a billing object.
    """
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text="ok")],
        stop_reason="end_turn",
        usage=usage,
    )


def _usage(
    *,
    input: int = 10,
    output: int = 20,
    cache_read: int = 0,
    cache_creation: int = 0,
) -> SimpleNamespace:
    """A minimal duck-typed Anthropic ``Message.usage``."""
    return SimpleNamespace(
        input_tokens=input,
        output_tokens=output,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_creation,
    )


def _http_response(status_code: int) -> httpx.Response:
    """A minimal ``httpx.Response`` for constructing API status errors."""
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return httpx.Response(status_code=status_code, request=request)


@pytest.fixture
def fake_anthropic():
    """Patch AsyncAnthropic so its client.messages.create is controllable."""
    instance = AsyncMock()
    instance.messages.create = AsyncMock(return_value=_message_response())
    with patch("src.agent.claude_client.AsyncAnthropic", return_value=instance):
        yield instance


class TestClaudeClient:
    async def test_success_returns_response(self, fake_anthropic) -> None:
        client = ClaudeClient(api_key="test-key")
        result = await client.send_messages([{"role": "user", "content": "hi"}])
        assert result.stop_reason == "end_turn"
        fake_messages = fake_anthropic.messages.create
        fake_messages.assert_awaited_once()
        kwargs = fake_messages.await_args.kwargs
        assert kwargs["model"] == "claude-sonnet-5-20250611"
        assert kwargs["max_tokens"] == 4096

    async def test_system_and_tools_passed_through(self, fake_anthropic) -> None:
        # prompt_caching=False keeps system/tools untouched (see TestPromptCaching).
        client = ClaudeClient(api_key="k", prompt_caching=False)
        tools = [{"name": "t"}]
        await client.send_messages(
            [{"role": "user", "content": "x"}],
            system="sys",
            tools=tools,
        )
        kwargs = fake_anthropic.messages.create.await_args.kwargs
        assert kwargs["system"] == "sys"
        assert kwargs["tools"] is tools

    async def test_retries_then_succeeds(self, fake_anthropic) -> None:
        # First call raises RateLimitError, second succeeds.
        fake_messages = fake_anthropic.messages.create
        fake_messages.side_effect = [
            RateLimitError("slow down", response=_http_response(429), body=None),
            _message_response(),
        ]
        client = ClaudeClient(api_key="k")

        result = await client.send_messages([])

        assert result.stop_reason == "end_turn"
        assert fake_messages.await_count >= 2

    async def test_retries_internal_server_error(self, fake_anthropic) -> None:
        fake_messages = fake_anthropic.messages.create
        fake_messages.side_effect = [
            InternalServerError("boom", response=_http_response(500), body=None),
            _message_response(),
        ]
        client = ClaudeClient(api_key="k")

        result = await client.send_messages([])

        assert result.stop_reason == "end_turn"
        assert fake_messages.await_count >= 2

    async def test_api_error_raises_claude_error(self, fake_anthropic) -> None:
        fake_messages = fake_anthropic.messages.create
        fake_messages.side_effect = APIError("bad request", request=None, body=None)
        client = ClaudeClient(api_key="k")

        with pytest.raises(ClaudeClientError, match="Anthropic API error"):
            await client.send_messages([])

    async def test_timeout_raises_claude_error(self, fake_anthropic) -> None:
        fake_messages = fake_anthropic.messages.create
        fake_messages.side_effect = APITimeoutError(request=None)
        client = ClaudeClient(api_key="k")

        with pytest.raises(ClaudeClientError, match="timed out"):
            await client.send_messages([])

    async def test_retries_exhausted_raises(self, fake_anthropic) -> None:
        fake_messages = fake_anthropic.messages.create
        fake_messages.side_effect = RateLimitError("slow", response=_http_response(429), body=None)
        client = ClaudeClient(api_key="k")

        with patch("src.agent.claude_client._MAX_RETRIES", 3):
            with pytest.raises(ClaudeClientError, match="after 3 retries"):
                await client.send_messages([])
        assert fake_messages.await_count == 3


class TestPromptCaching:
    """cache_control breakpoints on the system prompt and tool definitions."""

    async def test_system_string_becomes_cached_text_block(self, fake_anthropic) -> None:
        client = ClaudeClient(api_key="k")
        await client.send_messages([], system="hi")
        kwargs = fake_anthropic.messages.create.await_args.kwargs
        assert kwargs["system"] == [
            {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}
        ]

    async def test_system_block_list_caches_last_block_only(self, fake_anthropic) -> None:
        system = [
            {"type": "text", "text": "summary"},
            {"type": "text", "text": "base prompt"},
        ]
        client = ClaudeClient(api_key="k")
        await client.send_messages([], system=system)
        kwargs = fake_anthropic.messages.create.await_args.kwargs
        sent = kwargs["system"]
        assert "cache_control" not in sent[0]
        assert sent[1] == {
            "type": "text",
            "text": "base prompt",
            "cache_control": {"type": "ephemeral"},
        }
        # The caller's block list is never mutated.
        assert "cache_control" not in system[1]

    async def test_system_none_is_omitted(self, fake_anthropic) -> None:
        client = ClaudeClient(api_key="k")
        await client.send_messages([])
        kwargs = fake_anthropic.messages.create.await_args.kwargs
        assert "system" not in kwargs

    async def test_tools_mark_last_definition_only(self, fake_anthropic) -> None:
        tools = [{"name": "a"}, {"name": "b"}]
        client = ClaudeClient(api_key="k")
        await client.send_messages([], tools=tools)
        kwargs = fake_anthropic.messages.create.await_args.kwargs
        sent = kwargs["tools"]
        assert "cache_control" not in sent[0]
        assert sent[-1] == {"name": "b", "cache_control": {"type": "ephemeral"}}
        # The registry's shared tool defs are never mutated.
        assert "cache_control" not in tools[-1]

    async def test_empty_tools_unchanged(self, fake_anthropic) -> None:
        client = ClaudeClient(api_key="k")
        await client.send_messages([], tools=[])
        kwargs = fake_anthropic.messages.create.await_args.kwargs
        assert "tools" not in kwargs

    async def test_disabled_passes_through_unchanged(self, fake_anthropic) -> None:
        tools = [{"name": "t"}]
        client = ClaudeClient(api_key="k", prompt_caching=False)
        await client.send_messages([], system="sys", tools=tools)
        kwargs = fake_anthropic.messages.create.await_args.kwargs
        assert kwargs["system"] == "sys"
        assert kwargs["tools"] is tools

    async def test_streaming_marks_system_and_last_tool(self, fake_anthropic) -> None:
        fake_anthropic.messages.stream = MagicMock(
            return_value=_FakeStreamManager(_FakeStream(["x"]))
        )
        client = ClaudeClient(api_key="k")
        await client.stream_messages([], system="hi", tools=[{"name": "t"}])
        kwargs = fake_anthropic.messages.stream.call_args.kwargs
        assert kwargs["system"] == [
            {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}
        ]
        assert kwargs["tools"] == [{"name": "t", "cache_control": {"type": "ephemeral"}}]


# ---------------------------------------------------------------------------
# Streaming (messages.stream + text deltas)
# ---------------------------------------------------------------------------
class _FakeStream:
    """Duck-types anthropic's ``AsyncMessageStream``."""

    def __init__(
        self,
        deltas: list[str],
        *,
        mid_stream_error: Exception | None = None,
        final_usage: Any = None,
    ) -> None:
        self._deltas = deltas
        self._mid_stream_error = mid_stream_error
        self._final_usage = final_usage

    @property
    def text_stream(self) -> Any:
        async def _gen():
            for delta in self._deltas:
                yield delta
            if self._mid_stream_error is not None:
                raise self._mid_stream_error

        return _gen()

    async def get_final_message(self):
        return _message_response(usage=self._final_usage)


class _FakeStreamManager:
    """Duck-types anthropic's ``AsyncMessageStreamManager``."""

    def __init__(self, stream: Any = None, *, enter_error: Exception | None = None) -> None:
        self._stream = stream
        self._enter_error = enter_error

    async def __aenter__(self):
        if self._enter_error is not None:
            raise self._enter_error
        return self._stream

    async def __aexit__(self, *exc_info: object) -> bool:
        return False


class TestClaudeClientStreaming:
    async def test_streams_deltas_and_returns_final(self, fake_anthropic) -> None:
        fake_anthropic.messages.stream = MagicMock(
            return_value=_FakeStreamManager(_FakeStream(["Hel", "lo", " world"]))
        )
        deltas: list[str] = []

        async def sink(delta: str) -> None:
            deltas.append(delta)

        client = ClaudeClient(api_key="k")
        result = await client.stream_messages(
            [{"role": "user", "content": "hi"}],
            on_text_delta=sink,
        )

        assert deltas == ["Hel", "lo", " world"]
        assert result.stop_reason == "end_turn"
        kwargs = fake_anthropic.messages.stream.call_args.kwargs
        assert kwargs["model"] == "claude-sonnet-5-20250611"
        assert kwargs["max_tokens"] == 4096

    async def test_system_and_tools_passed_through(self, fake_anthropic) -> None:
        fake_anthropic.messages.stream = MagicMock(
            return_value=_FakeStreamManager(_FakeStream(["x"]))
        )
        tools = [{"name": "t"}]
        client = ClaudeClient(api_key="k", prompt_caching=False)
        await client.stream_messages([], system="sys", tools=tools)
        kwargs = fake_anthropic.messages.stream.call_args.kwargs
        assert kwargs["system"] == "sys"
        assert kwargs["tools"] is tools

    async def test_on_text_delta_none_works(self, fake_anthropic) -> None:
        fake_anthropic.messages.stream = MagicMock(
            return_value=_FakeStreamManager(_FakeStream(["text"]))
        )
        client = ClaudeClient(api_key="k")
        result = await client.stream_messages([{"role": "user", "content": "hi"}])
        assert result.stop_reason == "end_turn"

    async def test_retries_rate_limit_at_enter_then_succeeds(self, fake_anthropic) -> None:
        failing = _FakeStreamManager(
            enter_error=RateLimitError("slow down", response=_http_response(429), body=None)
        )
        good = _FakeStreamManager(_FakeStream(["ok"]))
        fake_anthropic.messages.stream = MagicMock(side_effect=[failing, good])
        deltas: list[str] = []

        async def sink(delta: str) -> None:
            deltas.append(delta)

        client = ClaudeClient(api_key="k")
        result = await client.stream_messages([], on_text_delta=sink)

        assert result.stop_reason == "end_turn"
        assert deltas == ["ok"]
        assert fake_anthropic.messages.stream.call_count == 2

    async def test_no_retry_after_delta_mid_stream_error(self, fake_anthropic) -> None:
        # A rate-limit *after* text has started streaming must not resend —
        # it would duplicate the already-emitted text.
        stream = _FakeStream(
            ["hi"],
            mid_stream_error=RateLimitError("boom", response=_http_response(429), body=None),
        )
        fake_anthropic.messages.stream = MagicMock(return_value=_FakeStreamManager(stream))
        deltas: list[str] = []

        async def sink(delta: str) -> None:
            deltas.append(delta)

        client = ClaudeClient(api_key="k")
        with pytest.raises(ClaudeClientError, match="mid-response"):
            await client.stream_messages([], on_text_delta=sink)

        assert deltas == ["hi"]
        assert fake_anthropic.messages.stream.call_count == 1

    async def test_retries_exhausted_raises(self, fake_anthropic) -> None:
        failing = _FakeStreamManager(
            enter_error=RateLimitError("slow", response=_http_response(429), body=None)
        )
        fake_anthropic.messages.stream = MagicMock(return_value=failing)
        client = ClaudeClient(api_key="k")

        with patch("src.agent.claude_client._MAX_RETRIES", 3):
            with pytest.raises(ClaudeClientError, match="after 3 retries"):
                await client.stream_messages([])
        assert fake_anthropic.messages.stream.call_count == 3

    async def test_timeout_raises(self, fake_anthropic) -> None:
        fake_anthropic.messages.stream = MagicMock(
            return_value=_FakeStreamManager(enter_error=APITimeoutError(request=None))
        )
        client = ClaudeClient(api_key="k")
        with pytest.raises(ClaudeClientError, match="timed out"):
            await client.stream_messages([])

    async def test_api_error_raises(self, fake_anthropic) -> None:
        fake_anthropic.messages.stream = MagicMock(
            return_value=_FakeStreamManager(
                enter_error=APIError("bad request", request=None, body=None)
            )
        )
        client = ClaudeClient(api_key="k")
        with pytest.raises(ClaudeClientError, match="Anthropic API error"):
            await client.stream_messages([])


class TestUsageAccumulation:
    """The shared client accumulates ``Usage`` across every LLM call."""

    async def test_usage_starts_empty(self, fake_anthropic) -> None:
        client = ClaudeClient(api_key="k")
        assert client.usage == Usage()

    async def test_send_messages_accumulates_and_counts_cache(self, fake_anthropic) -> None:
        fake_anthropic.messages.create.side_effect = [
            _message_response(usage=_usage(input=100, output=50, cache_read=10)),
            _message_response(usage=_usage(input=200, output=5, cache_creation=40)),
        ]
        client = ClaudeClient(api_key="k")
        await client.send_messages([])
        await client.send_messages([])
        assert client.usage == Usage(
            input_tokens=300,
            output_tokens=55,
            cache_read_input_tokens=10,
            cache_creation_input_tokens=40,
        )

    async def test_stream_accumulates(self, fake_anthropic) -> None:
        fake_anthropic.messages.stream = MagicMock(
            return_value=_FakeStreamManager(
                _FakeStream(["x"], final_usage=_usage(input=50, output=7))
            )
        )
        client = ClaudeClient(api_key="k")
        await client.stream_messages([])
        assert client.usage == Usage(input_tokens=50, output_tokens=7)

    async def test_missing_usage_leaves_unchanged(self, fake_anthropic) -> None:
        fake_anthropic.messages.create.side_effect = [
            _message_response(usage=_usage(input=5)),
            _message_response(),  # no usage attribute — must not crash
        ]
        client = ClaudeClient(api_key="k")
        await client.send_messages([])
        await client.send_messages([])
        # Only the first (billed) call contributed; the no-usage call added zero.
        assert client.usage == Usage(input_tokens=5, output_tokens=20)

    async def test_partial_usage_defaults_to_zero(self, fake_anthropic) -> None:
        fake_anthropic.messages.create.return_value = _message_response(
            usage=SimpleNamespace(input_tokens=10)
        )
        client = ClaudeClient(api_key="k")
        await client.send_messages([])
        assert client.usage == Usage(input_tokens=10)

    async def test_estimated_cost_reflects_model(self, fake_anthropic) -> None:
        client = ClaudeClient(api_key="k", model="claude-sonnet-5-20250611")
        fake_anthropic.messages.create.return_value = _message_response(
            usage=_usage(input=1_000_000, output=1_000_000)
        )
        await client.send_messages([])
        # sonnet list prices: $3/M in + $15/M out.
        assert client.estimated_cost() == pytest.approx(18.0)

    async def test_estimated_cost_zero_before_any_call(self, fake_anthropic) -> None:
        client = ClaudeClient(api_key="k")
        assert client.estimated_cost() == 0.0
