"""Thin wrapper around the Anthropic SDK."""

from __future__ import annotations

from typing import Any

from anthropic import (
    Anthropic,
    APIError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)


class ClaudeClientError(Exception):
    """Raised when a Claude API call fails after all retries."""


class ClaudeClient:
    """Handles the HTTP-level interaction with the Claude API."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-5-20250611",
        max_tokens: int = 4096,
    ) -> None:
        self._client = Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    # ------------------------------------------------------------------
    def send_messages(
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

        try:
            return self._client.messages.create(**kwargs)
        except RateLimitError:
            raise ClaudeClientError(
                "Rate limited by Anthropic API. Please wait and try again."
            )
        except APITimeoutError:
            raise ClaudeClientError(
                "Request to Anthropic API timed out. Check your network connection."
            )
        except InternalServerError:
            raise ClaudeClientError(
                "Anthropic API returned a server error. Please try again."
            )
        except APIError as exc:
            raise ClaudeClientError(
                f"Anthropic API error: {exc}"
            )
        except Exception as exc:
            raise ClaudeClientError(
                f"Unexpected error communicating with Anthropic API: {exc}"
            )
