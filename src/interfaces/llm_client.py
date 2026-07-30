"""Protocol for LLM backends (Anthropic, OpenAI, etc.)."""

from __future__ import annotations

from typing import Any, Protocol


class LLMClient(Protocol):
    """Interface for a chat LLM that supports tool use.

    The only method a client must provide is ``send_messages``, which
    accepts the same conversation structure that the orchestrator
    maintains and returns a response with ``content`` and ``stop_reason``
    attributes.
    """

    async def send_messages(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict] | None = None,
    ) -> Any:
        """Send a message list to the LLM and return the response.

        Args:
            messages: Conversation history in the Anthropic message format.
            system: Optional system prompt.
            tools: Optional list of tool definitions in Anthropic format.

        Returns:
            A response object with ``.content`` (list of blocks) and
            ``.stop_reason`` (str).
        """
        ...
