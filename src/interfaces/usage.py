"""Token-usage accounting for LLM calls (input, output, and cached tokens).

A single ``Usage`` is immutable and cumulative — the shared ``ClaudeClient``
accumulates one across every call (main agent, sub-agents, and summarizer),
and per-turn deltas are produced with ``__sub__``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: Estimated per-million-token USD rates, keyed by model-substring. These are
#: list prices and drift — the cost is an *estimate* used for the display and
#: the optional budget ceiling, never a bill.
_PRICING: dict[str, dict[str, float]] = {
    "opus": {
        "input": 15.0,
        "output": 75.0,
        "cache_write": 18.75,
        "cache_read": 1.50,
    },
    "sonnet": {
        "input": 3.0,
        "output": 15.0,
        "cache_write": 3.75,
        "cache_read": 0.30,
    },
    "haiku": {
        "input": 0.80,
        "output": 4.0,
        "cache_write": 1.0,
        "cache_read": 0.08,
    },
}
_DEFAULT_RATES = _PRICING["sonnet"]


def _rates_for(model: str | None) -> dict[str, float]:
    """Return the pricing table matching *model*, defaulting to sonnet."""
    if model:
        key = model.lower()
        for name, table in _PRICING.items():
            if name in key:
                return table
    return _DEFAULT_RATES


@dataclass(frozen=True)
class Usage:
    """Cumulative token usage for one or more LLM calls.

    ``cache_read_input_tokens`` and ``cache_creation_input_tokens`` are the
    Anthropic prompt-caching counters. Cached reads are billed at their own
    (much cheaper) rate, so they are *not* folded into ``input_tokens`` here —
    :meth:`total_input_tokens` counts all three so the reported cost reflects
    cache hits and writes.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    # ------------------------------------------------------------------
    @property
    def total_input_tokens(self) -> int:
        """All billed input: fresh plus cached reads and cache writes."""
        return self.input_tokens + self.cache_read_input_tokens + self.cache_creation_input_tokens

    @property
    def total_tokens(self) -> int:
        """Input-total plus output tokens."""
        return self.total_input_tokens + self.output_tokens

    # ------------------------------------------------------------------
    @classmethod
    def from_response(cls, response: Any) -> "Usage":
        """Extract usage from a duck-typed Anthropic ``Message``.

        A response without a ``usage`` attribute (test doubles, or a future
        provider that omits it) yields an empty ``Usage`` rather than an error.
        Missing individual counters default to zero.
        """
        usage = getattr(response, "usage", None)
        if usage is None:
            return cls()
        return cls(
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            cache_read_input_tokens=int(getattr(usage, "cache_read_input_tokens", 0) or 0),
            cache_creation_input_tokens=int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
        )

    def add(self, other: "Usage") -> "Usage":
        """Return a new ``Usage`` summing this and *other*."""
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_input_tokens=(self.cache_read_input_tokens + other.cache_read_input_tokens),
            cache_creation_input_tokens=(
                self.cache_creation_input_tokens + other.cache_creation_input_tokens
            ),
        )

    def __sub__(self, other: "Usage") -> "Usage":
        """Return the per-field difference, clamped at zero (turn deltas)."""
        return Usage(
            input_tokens=max(0, self.input_tokens - other.input_tokens),
            output_tokens=max(0, self.output_tokens - other.output_tokens),
            cache_read_input_tokens=max(
                0, self.cache_read_input_tokens - other.cache_read_input_tokens
            ),
            cache_creation_input_tokens=max(
                0, self.cache_creation_input_tokens - other.cache_creation_input_tokens
            ),
        )

    # ------------------------------------------------------------------
    def estimate_cost(self, model: str | None = None) -> float:
        """Estimated USD cost of these tokens at *model*'s per-token rates."""
        rates = _rates_for(model)
        per_million = 1_000_000
        return (
            self.input_tokens / per_million * rates["input"]
            + self.output_tokens / per_million * rates["output"]
            + self.cache_read_input_tokens / per_million * rates["cache_read"]
            + self.cache_creation_input_tokens / per_million * rates["cache_write"]
        )
