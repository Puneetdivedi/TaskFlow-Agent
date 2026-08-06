"""Shared token estimation for conversation memory and summarization."""

from __future__ import annotations


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token for English text).

    Shared by conversation pruning and semantic-memory consolidation so both
    use the same heuristic when deciding how much context to keep or drop.
    """
    return len(text) // 4
