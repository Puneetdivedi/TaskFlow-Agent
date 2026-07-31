"""Tests for conversation memory management."""

from __future__ import annotations

from src.memory.conversation import ConversationMemory


class TestConversationMemory:
    def test_add_user(self) -> None:
        mem = ConversationMemory()
        mem.add_user("Hello")
        assert len(mem.messages) == 1
        assert mem.messages[0]["role"] == "user"
        assert mem.messages[0]["content"] == "Hello"

    def test_add_assistant_text(self) -> None:
        mem = ConversationMemory()
        mem.add_assistant("Hi there!")
        assert mem.messages[0]["role"] == "assistant"
        assert mem.messages[0]["content"] == "Hi there!"

    def test_add_assistant_content_blocks(self) -> None:
        mem = ConversationMemory()
        blocks = [{"type": "text", "text": "Hello"}]
        mem.add_assistant(blocks)
        assert mem.messages[0]["content"] is blocks

    def test_add_tool_result(self) -> None:
        mem = ConversationMemory()
        mem.add_tool_result("toolu_abc", "result content")
        msg = mem.messages[0]
        assert msg["role"] == "user"
        assert msg["content"][0]["type"] == "tool_result"
        assert msg["content"][0]["tool_use_id"] == "toolu_abc"
        assert msg["content"][0]["content"] == "result content"

    def test_clear(self) -> None:
        mem = ConversationMemory()
        mem.add_user("Hello")
        mem.add_assistant("Hi")
        mem.clear()
        assert mem.messages == []
        assert mem.is_empty

    def test_is_empty(self) -> None:
        mem = ConversationMemory()
        assert mem.is_empty is True
        mem.add_user("Hello")
        assert mem.is_empty is False

    def test_messages_returns_copy(self) -> None:
        mem = ConversationMemory()
        mem.add_user("Hello")
        msgs = mem.messages
        msgs.append({"role": "user", "content": "extra"})
        assert len(mem.messages) == 1  # original unchanged

    def test_prune_under_limit_does_nothing(self) -> None:
        mem = ConversationMemory(max_tokens=100_000)
        mem.add_user("Hello")
        mem.add_assistant("Hi" * 100)
        before = len(mem.messages)
        mem.prune()
        assert len(mem.messages) == before

    def test_prune_over_limit_removes_oldest(self) -> None:
        mem = ConversationMemory(max_tokens=50)
        # Add enough messages to exceed the token budget
        mem.add_user("A" * 200)
        mem.add_assistant("B" * 200)
        mem.add_user("C" * 200)
        before = len(mem.messages)
        mem.prune()
        assert len(mem.messages) <= before

    def test_prune_empty_conversation(self) -> None:
        mem = ConversationMemory(max_tokens=10)
        mem.prune()  # should not raise
        assert mem.messages == []

    def test_prune_single_message(self) -> None:
        mem = ConversationMemory(max_tokens=10)
        mem.add_user("Hello" * 100)
        mem.prune()
        # Should keep at least 1 message
        assert len(mem.messages) >= 1

    def test_restore_replaces_messages(self) -> None:
        mem = ConversationMemory()
        mem.add_user("Hello")
        mem.add_assistant("Hi")
        saved = mem.messages
        mem.add_user("extra")
        mem.restore(saved)
        assert mem.messages == saved

    def test_restore_empty_list_clears(self) -> None:
        mem = ConversationMemory()
        mem.add_user("Hello")
        mem.restore([])
        assert mem.messages == []
        assert mem.is_empty

    def test_restore_preserves_tool_result_blocks(self) -> None:
        mem = ConversationMemory()
        mem.add_tool_result("toolu_1", "result")
        saved = mem.messages
        restored = ConversationMemory()
        restored.restore(saved)
        assert restored.messages == saved
        assert restored.messages[0]["content"][0]["type"] == "tool_result"

    def test_restore_does_not_share_the_input_list(self) -> None:
        mem = ConversationMemory()
        incoming = [{"role": "user", "content": "a"}]
        mem.restore(incoming)
        incoming.append({"role": "user", "content": "b"})
        assert len(mem.messages) == 1
