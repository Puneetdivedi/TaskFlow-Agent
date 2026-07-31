"""Tests for the CLI helper functions (non-interactive parts)."""

from __future__ import annotations

from src.ui.cli import print_banner, print_help, print_tools


class TestCLIHelpers:
    def test_print_banner_does_not_raise(self) -> None:
        print_banner()

    def test_print_help_does_not_raise(self) -> None:
        print_help()

    def test_print_tools_lists_registered_tools(self, mock_tool_registry) -> None:
        # mock_tool_registry only exposes "mock_tool"; build a minimal
        # stand-in orchestrator to exercise print_tools.
        class _Orch:
            tools = mock_tool_registry

        print_tools(_Orch())  # type: ignore[arg-type]
