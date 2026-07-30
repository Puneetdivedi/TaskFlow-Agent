"""Tests for the agent orchestrator (mocked client where possible)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.orchestrator import AgentOrchestrator, DEFAULT_SYSTEM_PROMPT
from src.tools.base import ToolError
from src.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------
class TestToolRegistry:
    def test_has_phase1_tools(self) -> None:
        registry = ToolRegistry()
        names = registry.tool_names
        assert "read_file" in names
        assert "write_file" in names
        assert "list_files" in names
        assert "search_files" in names
        assert "run_shell" in names
        assert "move_file" in names
        assert "delete_file" in names
        assert "file_index" in names

    def test_registry_safety_level_passed(self) -> None:
        registry = ToolRegistry(safety_level=0)
        with pytest.raises(ToolError, match="blocked at safety level 0"):
            import asyncio
            asyncio.run(registry.dispatch("run_shell", {"command": "echo hello"}))

    def test_registry_default_safety_allows_shell(self) -> None:
        """Safety level 1 (default) should allow shell commands."""
        import asyncio
        result = asyncio.run(ToolRegistry().dispatch("run_shell", {"command": "echo hello"}))
        assert "hello" in result

    def test_anthropic_defs_are_valid(self) -> None:
        registry = ToolRegistry()
        defs = registry.anthropic_tool_defs()
        for d in defs:
            assert "name" in d
            assert "description" in d
            assert "input_schema" in d
            assert "type" in d["input_schema"]
            assert "properties" in d["input_schema"]

    async def test_dispatch_unknown_tool(self) -> None:
        registry = ToolRegistry()
        with pytest.raises(ToolError, match="Unknown tool"):
            await registry.dispatch("nonexistent", {})


# ---------------------------------------------------------------------------
# AgentOrchestrator (no API call)
# ---------------------------------------------------------------------------
class TestAgentOrchestratorInit:
    def test_initializes_with_defaults(self) -> None:
        orch = AgentOrchestrator(api_key="test-key")
        assert orch._client is not None
        assert orch._system_prompt == DEFAULT_SYSTEM_PROMPT
        assert orch._max_tool_calls == 25

    def test_initializes_with_custom_prompt(self) -> None:
        custom = "You are a custom agent."
        orch = AgentOrchestrator(api_key="test-key", system_prompt=custom)
        assert orch._system_prompt == custom

    def test_tools_property(self) -> None:
        orch = AgentOrchestrator(api_key="test-key")
        assert isinstance(orch.tools, ToolRegistry)
