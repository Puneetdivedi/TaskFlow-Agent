"""Semantic memory — condensing old conversation turns into a rolling summary.

Long conversations are compacted instead of silently forgotten: turns above a
token budget are summarized by the LLM and injected back into the model's
context by the orchestrator. The summary is stored on the memory object
(``IMemory.summary``) and can be persisted per session.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.interfaces import LLMClient
from src.memory.tokens import estimate_tokens

logger = logging.getLogger(__name__)


class SummaryError(Exception):
    """Raised when generating a conversation summary fails.

    The orchestrator catches this and continues with the pre-consolidation
    state — a failed summary never crashes a turn.
    """


#: System prompt for the summary model. Asks for a dense, fact-preserving
#: running summary — names, dates, decisions, preferences, file paths and
#: numbers — not opinions or flavor.
_SUMMARIZER_PROMPT = """\
You are a memory system for an AI assistant. You read older parts of a
conversation and produce a concise running summary that lets the assistant
continue as if it remembered everything.

Preserve concrete facts the user might rely on later: names, dates, deadlines,
decisions, preferences, file paths, URLs, numbers, and completed actions. Drop
small talk and re-derivable details. Use compact prose or bullets. Keep it
under ~500 words. Never invent facts."""


def format_turns(messages: list[dict[str, Any]]) -> str:
    """Render message dicts into a compact text block for the summarizer.

    Each message becomes ``ROLE: content``; content blocks (tool results, tool
    calls) are JSON-encoded so nothing is lost.
    """
    lines: list[str] = []
    for msg in messages:
        role = msg.get("role", "message")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = json.dumps(content, ensure_ascii=False)
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _is_plain_user(msg: dict[str, Any]) -> bool:
    """True for a user message that starts a fresh turn (not a tool result)."""
    if msg.get("role") != "user":
        return False
    content = msg.get("content")
    if isinstance(content, list):
        return not any(isinstance(c, dict) and c.get("type") == "tool_result" for c in content)
    return True


def plan_consolidation(
    messages: list[dict[str, Any]],
    *,
    keep_recent_tokens: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split *messages* into ``(to_summarize, keep)``.

    Walks from the *end* backward, accumulating the most recent context until
    roughly *keep_recent_tokens* are kept, and stops at a *turn boundary* — the
    newest kept message is a plain user message, so neither half ends up with a
    dangling exchange and trailing ``tool_result`` messages stay with the
    assistant turn that produced them. The older prefix becomes
    ``to_summarize``. If nothing is safely summarizable (fewer than 2 messages,
    or the whole conversation fits), returns ``([], messages)``.
    """
    if len(messages) < 2:
        return [], messages

    tokens = 0
    keep_rev: list[dict[str, Any]] = []

    for msg in reversed(messages):
        keep_rev.append(msg)
        tokens += estimate_tokens(str(msg))
        if tokens >= keep_recent_tokens and _is_plain_user(msg):
            break

    if len(keep_rev) == len(messages):
        # The whole conversation was kept — nothing older to summarize.
        return [], messages

    kept = list(reversed(keep_rev))
    to_summarize = messages[: len(messages) - len(kept)]
    return to_summarize, kept


class ConversationSummarizer:
    """Compresses conversation turns into a running summary via the LLM."""

    def __init__(
        self,
        client: LLMClient,
        *,
        system_prompt: str = _SUMMARIZER_PROMPT,
        max_tokens: int = 800,
    ) -> None:
        self._client = client
        self._system_prompt = system_prompt
        self._max_tokens = max_tokens

    async def summarize(self, turn_text: str, existing: str | None = None) -> str:
        """Return a merged summary of *turn_text* and any *existing* summary.

        Raises :class:`SummaryError` if the LLM call fails; returns *existing*
        (or ``""``) if the model produces no text.
        """
        parts: list[str] = [turn_text.strip()]
        if existing:
            parts.insert(
                0,
                f"Existing summary (extend or replace it, keeping prior facts):\n{existing}",
            )
        payload = "\n\n".join(parts)
        messages = [{"role": "user", "content": payload}]

        try:
            response = await self._client.send_messages(
                messages=messages,
                system=self._system_prompt,
                tools=None,
            )
        except Exception as exc:
            raise SummaryError(f"Failed to summarize conversation: {exc}") from exc

        text = _extract_text(response)
        if not text.strip():
            return (existing or "").strip()
        return text.strip()


def _extract_text(response: Any) -> str:
    """Join text blocks from a duck-typed Anthropic ``Message``."""
    parts: list[str] = []
    for block in getattr(response, "content", []):
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts)
