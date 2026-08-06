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


def _message_response() -> SimpleNamespace:
    """A minimal duck-typed Anthropic ``Message``."""
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text="ok")],
        stop_reason="end_turn",
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
        client = ClaudeClient(api_key="k")
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
    ) -> None:
        self._deltas = deltas
        self._mid_stream_error = mid_stream_error

    @property
    def text_stream(self) -> Any:
        async def _gen():
            for delta in self._deltas:
                yield delta
            if self._mid_stream_error is not None:
                raise self._mid_stream_error

        return _gen()

    async def get_final_message(self):
        return _message_response()


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
        client = ClaudeClient(api_key="k")
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
