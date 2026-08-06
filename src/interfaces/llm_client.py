"""Protocol for LLM backends (Anthropic, OpenAI, etc.)."""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Protocol, TypeAlias

#: Async callback receiving each chunk of assistant text as it is generated.
TextDeltaSink: TypeAlias = Callable[[str], Awaitable[None]]
#: Async callback receiving each tool call (name, arguments) before it runs.
ToolCallSink: TypeAlias = Callable[[str, dict[str, Any]], Awaitable[None]]
#: The ``system`` argument: a plain prompt string or a list of text blocks
#: (used by the orchestrator to inject a rolling conversation summary ahead of
#: the base prompt).
SystemParam: TypeAlias = str | list[dict[str, str]] | None


class LLMClient(Protocol):
    """Interface for a chat LLM that supports tool use.

    The only method a client must provide is ``send_messages``, which
    accepts the same conversation structure that the orchestrator
    maintains and returns a response with ``content`` and ``stop_reason``
    attributes. Clients may additionally implement ``stream_messages`` to
    deliver text incrementally.
    """

    async def send_messages(
        self,
        messages: list[dict[str, Any]],
        system: SystemParam = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        """Send a message list to the LLM and return the response.

        Args:
            messages: Conversation history in the Anthropic message format.
            system: Optional system prompt (string or text-block list).
            tools: Optional list of tool definitions in Anthropic format.

        Returns:
            A response object with ``.content`` (list of blocks) and
            ``.stop_reason`` (str).
        """
        ...

    async def stream_messages(
        self,
        messages: list[dict[str, Any]],
        system: SystemParam = None,
        tools: list[dict[str, Any]] | None = None,
        *,
        on_text_delta: TextDeltaSink | None = None,
    ) -> Any:
        """Like :meth:`send_messages`, but calls *on_text_delta* with each
        text chunk as it arrives.

        Returns the same full response object as ``send_messages``.
        """
        ...
