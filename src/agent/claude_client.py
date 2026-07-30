"""Thin wrapper around the (async) Anthropic SDK."""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any

logger = logging.getLogger(__name__)

from anthropic import (
    AsyncAnthropic,
    APIError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)

_RETRYABLE_STATUSES = (RateLimitError, InternalServerError)
_MAX_RETRIES = 5
_BASE_DELAY = 1.0
_MAX_DELAY = 16.0


class ClaudeClientError(Exception):
    """Raised when a Claude API call fails after all retries."""


class ClaudeClient:
    """Handles the HTTP-level interaction with the Claude API (async).

    Automatically retries on rate limits and server errors with
    exponential backoff + jitter (up to :const:`_MAX_RETRIES` attempts).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-5-20250611",
        max_tokens: int = 4096,
    ) -> None:
        self._client = AsyncAnthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    # ------------------------------------------------------------------
    async def send_messages(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict] | None = None,
    ) -> Any:
        """Send a message list to Claude and return *the raw response object*.

        The caller (orchestrator) is responsible for inspecting
        ``response.stop_reason`` and ``response.content``.

        Raises ClaudeClientError on API failures (rate limits, auth,
        server errors, timeouts) — never the raw Anthropic exception.
        """
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        logger.debug("Calling Claude API — model=%s %d messages", self._model, len(messages))
        return await self._call_with_retry(kwargs)

    # ------------------------------------------------------------------
    async def _call_with_retry(self, kwargs: dict[str, Any]) -> Any:
        """Call ``messages.create`` with exponential backoff retry."""
        last_exc: Exception | None = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                resp = await self._client.messages.create(**kwargs)
                logger.debug("Claude API responded — stop_reason=%s", resp.stop_reason)
                return resp
            except _RETRYABLE_STATUSES as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    delay = min(_BASE_DELAY * 2**attempt + random.uniform(0, 1), _MAX_DELAY)
                    logger.warning(
                        "Claude API retryable error (attempt %d/%d): %s. "
                        "Retrying in %.1fs…",
                        attempt, _MAX_RETRIES, exc.__class__.__name__, delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "Claude API failed after %d retries: %s",
                        _MAX_RETRIES, exc,
                    )
            except APITimeoutError:
                raise ClaudeClientError(
                    "Request to Anthropic API timed out. Check your network connection."
                )
            except APIError as exc:
                raise ClaudeClientError(
                    f"Anthropic API error: {exc}"
                )
            except Exception as exc:
                raise ClaudeClientError(
                    f"Unexpected error communicating with Anthropic API: {exc}"
                )

        # All retries exhausted
        raise ClaudeClientError(
            f"Anthropic API failed after {_MAX_RETRIES} retries: {last_exc}"
        ) from last_exc
