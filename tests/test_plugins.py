"""Tests for plugin discovery."""

from __future__ import annotations

from src.plugins import discover_tools


class TestPluginDiscovery:
    def test_discovers_no_plugins_in_clean_env(self) -> None:
        """Without any installed plugins, discovery returns an empty list."""
        tools = discover_tools()
        assert isinstance(tools, list)
        assert len(tools) == 0
