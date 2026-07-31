"""Tests for the ClaudeClient API wrapper (retries and error handling)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

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
