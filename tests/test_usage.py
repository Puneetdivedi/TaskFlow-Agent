"""Tests for the ``Usage`` value object (token accounting and cost estimates)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.interfaces.usage import Usage


class TestUsageDefaults:
    def test_empty_usage_is_all_zero(self) -> None:
        usage = Usage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.cache_read_input_tokens == 0
        assert usage.cache_creation_input_tokens == 0
        assert usage.total_input_tokens == 0
        assert usage.total_tokens == 0

    def test_total_input_counts_fresh_plus_cached(self) -> None:
        usage = Usage(input_tokens=10, cache_read_input_tokens=3, cache_creation_input_tokens=2)
        assert usage.total_input_tokens == 15

    def test_total_tokens_includes_output(self) -> None:
        usage = Usage(input_tokens=10, output_tokens=4)
        assert usage.total_tokens == 14


class TestFromResponse:
    def test_no_usage_attribute_yields_empty(self) -> None:
        response = SimpleNamespace(content=[], stop_reason="end_turn")
        assert Usage.from_response(response) == Usage()

    def test_full_usage_extracted(self) -> None:
        response = SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=100,
                output_tokens=50,
                cache_read_input_tokens=10,
                cache_creation_input_tokens=5,
            )
        )
        assert Usage.from_response(response) == Usage(
            input_tokens=100,
            output_tokens=50,
            cache_read_input_tokens=10,
            cache_creation_input_tokens=5,
        )

    def test_partial_usage_defaults_missing_to_zero(self) -> None:
        response = SimpleNamespace(usage=SimpleNamespace(input_tokens=7))
        assert Usage.from_response(response) == Usage(input_tokens=7)

    def test_usage_none_member_is_zero(self) -> None:
        response = SimpleNamespace(usage=SimpleNamespace(input_tokens=None, output_tokens=None))
        assert Usage.from_response(response) == Usage()


class TestAddSubtract:
    def test_add_sums_fields(self) -> None:
        a = Usage(input_tokens=10, output_tokens=2)
        b = Usage(input_tokens=5, cache_read_input_tokens=3)
        assert a.add(b) == Usage(
            input_tokens=15,
            output_tokens=2,
            cache_read_input_tokens=3,
        )

    def test_add_is_non_mutating(self) -> None:
        a = Usage(input_tokens=10)
        a.add(Usage(input_tokens=5))
        assert a == Usage(input_tokens=10)

    def test_subtract_gives_turn_delta(self) -> None:
        before = Usage(input_tokens=10, output_tokens=2)
        after = Usage(input_tokens=25, output_tokens=8, cache_read_input_tokens=4)
        assert after - before == Usage(
            input_tokens=15,
            output_tokens=6,
            cache_read_input_tokens=4,
        )

    def test_subtract_clamps_at_zero(self) -> None:
        before = Usage(input_tokens=100, cache_read_input_tokens=5)
        after = Usage(input_tokens=40)
        assert after - before == Usage(input_tokens=0, cache_read_input_tokens=0)


class TestEstimateCost:
    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("claude-sonnet-5-20250611", 18.0),  # $3/M in + $15/M out
            ("claude-opus-5", 90.0),  # $15/M in + $75/M out
            ("claude-haiku-4-5", 4.8),  # $0.80/M in + $4/M out
        ],
    )
    def test_million_tokens_each(self, model: str, expected: float) -> None:
        usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
        assert usage.estimate_cost(model) == pytest.approx(expected)

    def test_cache_rates_are_cheaper(self) -> None:
        usage = Usage(cache_read_input_tokens=1_000_000)
        # Sonnet cached-read is $0.30/M.
        assert usage.estimate_cost("claude-sonnet-5-20250611") == pytest.approx(0.30)

    def test_unknown_model_falls_back_to_sonnet(self) -> None:
        usage = Usage(input_tokens=1_000_000)
        assert usage.estimate_cost("claude-future-9") == pytest.approx(3.0)

    def test_no_model_uses_default(self) -> None:
        usage = Usage(input_tokens=1_000_000)
        assert usage.estimate_cost(None) == pytest.approx(3.0)

    def test_empty_is_free(self) -> None:
        assert Usage().estimate_cost("claude-opus-5") == 0.0
