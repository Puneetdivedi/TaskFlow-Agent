"""Conversation history management — append, prune, and format for the LLM."""

from __future__ import annotations

from typing import Any


class ConversationMemory:
    """In-memory conversation history with token-aware pruning."""

    def __init__(self, max_tokens: int = 100_000) -> None:
        self._messages: list[dict[str, Any]] = []
        self._max_tokens = max_tokens

    # ------------------------------------------------------------------
    @property
    def messages(self) -> list[dict[str, Any]]:
        return list(self._messages)

    @property
    def is_empty(self) -> bool:
        return len(self._messages) == 0

    # ------------------------------------------------------------------
    def add_user(self, content: str) -> None:
        self._messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str | list[dict]) -> None:
        self._messages.append({"role": "assistant", "content": content})

    def add_tool_result(self, tool_use_id: str, content: str) -> None:
        self._messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": content,
                    }
                ],
            }
        )

    def add_message(self, msg: dict[str, Any]) -> None:
        self._messages.append(msg)

    # ------------------------------------------------------------------
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate (~4 chars per token for English text)."""
        return len(text) // 4

    def prune(self) -> None:
        """Drop oldest user/assistant pairs while staying under the token limit.

        Removes complete exchanges (user → assistant → tool_results)
        from the front until the estimated token count fits within the limit.
        The most recent message(s) are always preserved.
        """
        if self._estimate_tokens(str(self._messages)) <= self._max_tokens:
            return

        # Strategy: count how many messages to drop from the front,
        # removing whole user↔assistant rounds (including any trailing
        # tool_result messages that belong to the dropped assistant turn).
        while len(self._messages) > 2:
            idx = 0
            # Skip past any tool_result messages (they belong to the
            # assistant turn that was already removed in a prior cycle).
            while idx < len(self._messages) and self._messages[idx].get("role") == "user":
                content = self._messages[idx].get("content", "")
                if isinstance(content, list) and any(
                    isinstance(c, dict) and c.get("type") == "tool_result" for c in content
                ):
                    idx += 1
                else:
                    break

            if idx >= len(self._messages) - 1:
                # Nothing left to prune safely — bail out.
                break

            # Remove this user message + its assistant response.
            # (After removal, any tool_result messages that follow the
            #  removed assistant will be cleaned up on the next iteration.)
            self._messages.pop(idx)      # user message
            if idx < len(self._messages) and self._messages[idx].get("role") == "assistant":
                self._messages.pop(idx)  # assistant response

            if self._estimate_tokens(str(self._messages)) <= self._max_tokens:
                break

    def clear(self) -> None:
        self._messages.clear()
