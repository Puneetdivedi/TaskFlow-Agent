"""Thin wrapper around the (async) Anthropic SDK."""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any

from anthropic import (
    APIError,
    APITimeoutError,
    AsyncAnthropic,
    InternalServerError,
    RateLimitError,
)

from src.interfaces.llm_client import SystemParam, TextDeltaSink
from src.interfaces.usage import Usage

logger = logging.getLogger(__name__)

_RETRYABLE_STATUSES = (RateLimitError, InternalServerError)
_MAX_RETRIES = 5
_BASE_DELAY = 1.0
_MAX_DELAY = 16.0

#: Anthropic prompt-caching marker. Attaching this to the last system text
#: block and the last tool definition caches everything up to and including
#: that block for 5 minutes, so unchanged prefixes are not re-encoded on every
#: turn. The API silently ignores markers on prefixes below the minimum
#: cacheable size (~1024 tokens), so annotating small prompts is harmless.
_CACHE_CONTROL: dict[str, str] = {"type": "ephemeral"}


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
        *,
        prompt_caching: bool = True,
    ) -> None:
        self._client = AsyncAnthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens
        self._prompt_caching = prompt_caching
        #: Cumulative token usage across every call this client makes. Because
        #: the DI container shares one ClaudeClient between the orchestrator,
        #: sub-agents, and the summarizer, this captures the whole process.
        self.usage = Usage()

    # ------------------------------------------------------------------
    def estimated_cost(self) -> float:
        """Estimated USD cost of the tokens accumulated in :attr:`usage`.

        The model is known here, so callers never have to supply it — this is
        what the orchestrator's budget ceiling and the CLI display read.
        """
        return self.usage.estimate_cost(self._model)

    # ------------------------------------------------------------------
    def _maybe_cache_system(self, system: SystemParam) -> SystemParam:
        """Attach a cache breakpoint to the *last* system block.

        A plain string is converted to a single ``{"type": "text", ...}``
        block so it can carry ``cache_control``. A block list gets the marker
        on its final block only (the base prompt), so the whole prefix —
        summary + base prompt — is cached together. The caller's list is never
        mutated.
        """
        if not self._prompt_caching or not system:
            return system
        if isinstance(system, str):
            return [{"type": "text", "text": system, "cache_control": _CACHE_CONTROL}]
        blocks = [dict(block) for block in system]
        if blocks:
            blocks[-1] = {**blocks[-1], "cache_control": _CACHE_CONTROL}
        return blocks

    def _maybe_cache_tools(
        self,
        tool_defs: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]] | None:
        """Attach a cache breakpoint to the *last* tool definition.

        Returns a shallow copy so the registry's shared descriptors are never
        mutated. Marking the final tool caches the system blocks and the whole
        tool list as one prefix.
        """
        if not self._prompt_caching or not tool_defs:
            return tool_defs
        defs = [dict(defn) for defn in tool_defs]
        defs[-1] = {**defs[-1], "cache_control": _CACHE_CONTROL}
        return defs

    # ------------------------------------------------------------------
    async def send_messages(
        self,
        messages: list[dict[str, Any]],
        system: SystemParam = None,
        tools: list[dict[str, Any]] | None = None,
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
            kwargs["system"] = self._maybe_cache_system(system)
        if tools:
            kwargs["tools"] = self._maybe_cache_tools(tools)

        logger.debug("Calling Claude API — model=%s %d messages", self._model, len(messages))
        response = await self._call_with_retry(kwargs)
        self.usage = self.usage.add(Usage.from_response(response))
        return response

    # ------------------------------------------------------------------
    async def stream_messages(
        self,
        messages: list[dict[str, Any]],
        system: SystemParam = None,
        tools: list[dict[str, Any]] | None = None,
        *,
        on_text_delta: TextDeltaSink | None = None,
    ) -> Any:
        """Like :meth:`send_messages`, but streams text through *on_text_delta*.

        Each text delta is awaited through the callback as it arrives. The
        returned object is the full ``Message`` — same ``.content`` blocks
        and ``.stop_reason`` as a non-streaming call.
        """
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = self._maybe_cache_system(system)
        if tools:
            kwargs["tools"] = self._maybe_cache_tools(tools)

        logger.debug(
            "Calling Claude API (streaming) — model=%s %d messages",
            self._model,
            len(messages),
        )
        response = await self._stream_with_retry(kwargs, on_text_delta=on_text_delta)
        self.usage = self.usage.add(Usage.from_response(response))
        return response

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
                        "Claude API retryable error (attempt %d/%d): %s. Retrying in %.1fs…",
                        attempt,
                        _MAX_RETRIES,
                        exc.__class__.__name__,
                        delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "Claude API failed after %d retries: %s",
                        _MAX_RETRIES,
                        exc,
                    )
            except APITimeoutError:
                raise ClaudeClientError(
                    "Request to Anthropic API timed out. Check your network connection."
                )
            except APIError as exc:
                raise ClaudeClientError(f"Anthropic API error: {exc}")
            except Exception as exc:
                raise ClaudeClientError(f"Unexpected error communicating with Anthropic API: {exc}")

        # All retries exhausted
        raise ClaudeClientError(
            f"Anthropic API failed after {_MAX_RETRIES} retries: {last_exc}"
        ) from last_exc

    # ------------------------------------------------------------------
    async def _stream_with_retry(
        self,
        kwargs: dict[str, Any],
        *,
        on_text_delta: TextDeltaSink | None,
    ) -> Any:
        """Stream ``messages.stream`` with exponential backoff retry.

        Retries (rate limits, server errors) are only attempted while no text
        has been emitted yet. Once a delta has reached the caller we can
        never resend without duplicating output, so a mid-stream failure
        raises immediately.
        """
        started = False
        last_exc: Exception | None = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                # messages.stream() returns its manager synchronously; the HTTP
                # request fires on __aenter__, where rate-limit/5xx errors land.
                manager = self._client.messages.stream(**kwargs)
                async with manager as stream:
                    if on_text_delta is not None:
                        async for delta in stream.text_stream:
                            started = True
                            await on_text_delta(delta)
                    final = await stream.get_final_message()
                logger.debug("Claude API streamed response — stop_reason=%s", final.stop_reason)
                return final
            except _RETRYABLE_STATUSES as exc:
                last_exc = exc
                if started or attempt >= _MAX_RETRIES:
                    break
                delay = min(_BASE_DELAY * 2**attempt + random.uniform(0, 1), _MAX_DELAY)
                logger.warning(
                    "Claude API stream retryable error (attempt %d/%d): %s. Retrying in %.1fs…",
                    attempt,
                    _MAX_RETRIES,
                    exc.__class__.__name__,
                    delay,
                )
                await asyncio.sleep(delay)
            except APITimeoutError:
                raise ClaudeClientError(
                    "Request to Anthropic API timed out. Check your network connection."
                )
            except APIError as exc:
                raise ClaudeClientError(f"Anthropic API error: {exc}")
            except Exception as exc:
                raise ClaudeClientError(f"Unexpected error communicating with Anthropic API: {exc}")

        if started:
            raise ClaudeClientError("Anthropic API stream failed mid-response after emitting text.")
        raise ClaudeClientError(
            f"Anthropic API failed after {_MAX_RETRIES} retries: {last_exc}"
        ) from last_exc
