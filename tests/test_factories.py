"""Tests for the production dependency-injection wiring."""

from __future__ import annotations

from pathlib import Path

import pytest

from config.settings import Settings
from src.agent.orchestrator import AgentOrchestrator
from src.di.container import DIContainer
from src.di.factories import create_production_orchestrator, register_defaults
from src.interfaces import IMemory, IToolRegistry, LLMClient
from src.memory.conversation import ConversationMemory
from src.tools.registry import ToolRegistry


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    return Settings(
        anthropic_api_key="test-key",
        work_dir=tmp_path,
        memory_dir=tmp_path / "memory",
    )


class TestRegisterDefaults:
    def test_registers_all_interfaces(self, test_settings: Settings) -> None:
        container = DIContainer()
        register_defaults(container, test_settings)

        assert isinstance(container.resolve(LLMClient), object)
        assert isinstance(container.resolve(IToolRegistry), ToolRegistry)
        assert isinstance(container.resolve(IMemory), ConversationMemory)

    def test_registry_receives_work_dir_and_safety(self, test_settings: Settings) -> None:
        container = DIContainer()
        register_defaults(container, test_settings)

        registry: ToolRegistry = container.resolve(IToolRegistry)
        assert registry.tool_names  # non-empty
        shell_tool = registry._tools["run_shell"]
        assert shell_tool._safety_level == test_settings.safety_level

    def test_resolve_is_singleton(self, test_settings: Settings) -> None:
        container = DIContainer()
        register_defaults(container, test_settings)

        assert container.resolve(IMemory) is container.resolve(IMemory)


class TestCreateProductionOrchestrator:
    def test_returns_wired_orchestrator(self, test_settings: Settings) -> None:
        orch = create_production_orchestrator(settings=test_settings)
        assert isinstance(orch, AgentOrchestrator)
        assert orch.tools.tool_names
        assert isinstance(orch.memory, ConversationMemory)
        assert orch._max_tool_calls == test_settings.max_tool_calls_per_turn
