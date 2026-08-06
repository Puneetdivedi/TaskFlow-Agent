"""Tests for semantic memory — token estimation, compaction planning, summarizer."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.memory.summary import (
    ConversationSummarizer,
    SummaryError,
    format_turns,
    plan_consolidation,
)
from src.memory.tokens import estimate_tokens


class TestEstimateTokens:
    def test_empty(self) -> None:
        assert estimate_tokens("") == 0

    def test_approx_four_chars_per_token(self) -> None:
        assert estimate_tokens("abcd") == 1
        assert estimate_tokens("a" * 40) == 10

    def test_whitespace_counts(self) -> None:
        assert estimate_tokens(" " * 8) == 2


class TestFormatTurns:
    def test_renders_roles_and_content(self) -> None:
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        text = format_turns(messages)
        assert "user: hi" in text
        assert "assistant: hello" in text

    def test_json_encodes_content_blocks(self) -> None:
        messages = [
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}],
            }
        ]
        text = format_turns(messages)
        assert '"type": "tool_result"' in text
        assert '"tool_use_id": "t1"' in text


def _exchange(user: str, assistant: str) -> list[dict]:
    """A plain user→assistant pair (each ~100 chars)."""
    return [
        {"role": "user", "content": user * 100},
        {"role": "assistant", "content": assistant * 100},
    ]


class TestPlanConsolidation:
    def test_empty_and_single(self) -> None:
        assert plan_consolidation([], keep_recent_tokens=1) == ([], [])
        one = [{"role": "user", "content": "x"}]
        assert plan_consolidation(one, keep_recent_tokens=1) == ([], one)

    def test_tiny_conversation_all_kept(self) -> None:
        messages = _exchange("u", "a")
        assert plan_consolidation(messages, keep_recent_tokens=1) == ([], messages)

    def test_splits_at_plain_user_boundary(self) -> None:
        # Old prefix is summarizable; the newest exchange is kept intact.
        messages = (
            _exchange("a", "b")
            + _exchange("c", "d")
            + _exchange("e", "f")  # the "recent" exchange, never cut
        )
        to_summarize, kept = plan_consolidation(messages, keep_recent_tokens=1)
        assert to_summarize == _exchange("a", "b") + _exchange("c", "d")
        assert kept == _exchange("e", "f")

    def test_keeps_tool_results_with_their_turn(self) -> None:
        # A trailing tool_result belongs to the *older* assistant turn, so it
        # is summarized together with it, never split across the boundary.
        messages = _exchange("x", "y") + [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "t1", "name": "ls", "input": {}}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}],
            },
            {"role": "user", "content": "z" * 100},  # newest plain user
            {"role": "assistant", "content": "w" * 100},
        ]
        to_summarize, kept = plan_consolidation(messages, keep_recent_tokens=1)
        # The tool exchange (tool_use + tool_result) stays with the summarized
        # half alongside its assistant turn, and the newest pair is kept whole.
        assert to_summarize == _exchange("x", "y") + [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "t1", "name": "ls", "input": {}}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}],
            },
        ]
        assert kept == [
            {"role": "user", "content": "z" * 100},
            {"role": "assistant", "content": "w" * 100},
        ]

    def test_respects_token_bound(self) -> None:
        # A large keep_recent_tokens that the conversation never reaches means
        # everything is kept (nothing older to summarize).
        messages = _exchange("a", "b")  # ~50 tokens total, well under 5000
        assert plan_consolidation(messages, keep_recent_tokens=5000) == ([], messages)


class _StubLLM:
    """LLMClient stub that records calls and returns a canned response."""

    def __init__(self, text: str = "", *, error: Exception | None = None) -> None:
        self.text = text
        self.error = error
        self.sent_messages: list[list[dict]] = []
        self.sent_system: list[str | list[dict]] = []

    async def send_messages(
        self,
        messages: list[dict] | None = None,
        system: str | None = None,
        tools: list[dict] | None = None,
    ) -> object:
        self.sent_messages.append(list(messages or []))
        self.sent_system.append(system)  # type: ignore[arg-type]
        if self.error is not None:
            raise self.error
        content = [] if not self.text else [SimpleNamespace(type="text", text=self.text)]
        return SimpleNamespace(content=content, stop_reason="end_turn")


class TestConversationSummarizer:
    async def test_returns_model_text(self) -> None:
        client = _StubLLM(text="ROLLED UP")
        summarizer = ConversationSummarizer(client)  # type: ignore[arg-type]

        assert await summarizer.summarize("turn text") == "ROLLED UP"
        assert client.sent_messages == [[{"role": "user", "content": "turn text"}]]
        assert isinstance(client.sent_system[0], str)

    async def test_merges_existing_summary_into_payload(self) -> None:
        client = _StubLLM(text="new")
        summarizer = ConversationSummarizer(client)  # type: ignore[arg-type]

        await summarizer.summarize("new turns", existing="old summary")
        payload = client.sent_messages[0][0]["content"]
        assert "old summary" in payload
        assert "new turns" in payload

    async def test_empty_response_keeps_existing(self) -> None:
        client = _StubLLM(text="")
        summarizer = ConversationSummarizer(client)  # type: ignore[arg-type]

        assert await summarizer.summarize("new turns", existing="old") == "old"
        assert await summarizer.summarize("new turns") == ""

    async def test_api_failure_raises_summary_error(self) -> None:
        client = _StubLLM(error=RuntimeError("boom"))
        summarizer = ConversationSummarizer(client)  # type: ignore[arg-type]

        with pytest.raises(SummaryError):
            await summarizer.summarize("turn text")
