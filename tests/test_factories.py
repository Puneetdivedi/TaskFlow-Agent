"""Tests for the production dependency-injection wiring."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from config.settings import Settings
from src.agent.claude_client import ClaudeClient
from src.agent.orchestrator import AgentOrchestrator
from src.di.container import DIContainer
from src.di.factories import (
    AppComponents,
    create_production_app,
    create_production_orchestrator,
    register_defaults,
)
from src.interfaces import IMemory, ISessionStore, ITaskStore, IToolRegistry, LLMClient
from src.memory.conversation import ConversationMemory
from src.memory.session_store import SessionStore
from src.memory.task_store import TaskStore
from src.tools.base import Tool, ToolError
from src.tools.registry import ToolRegistry


class _EchoTool(Tool):
    """A minimal plugin tool used to verify the discovery wiring."""

    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "Echoes the given text back"

    @property
    def input_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {"text": {"type": "string"}}}

    async def run(self, **kwargs: Any) -> str:
        return f"Echo: {kwargs.get('text', '')}"


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    return Settings(
        anthropic_api_key="test-key",
        work_dir=tmp_path,
        memory_dir=tmp_path / "memory",
        db_path=tmp_path / "taskflow.db",
    )


class TestRegisterDefaults:
    def test_registers_all_interfaces(self, test_settings: Settings) -> None:
        container = DIContainer()
        register_defaults(container, test_settings)

        assert isinstance(container.resolve(LLMClient), object)
        assert isinstance(container.resolve(IToolRegistry), ToolRegistry)
        assert isinstance(container.resolve(IMemory), ConversationMemory)
        assert isinstance(container.resolve(ISessionStore), SessionStore)
        assert isinstance(container.resolve(ITaskStore), TaskStore)

    def test_session_store_uses_configured_db_path(self, test_settings: Settings) -> None:
        container = DIContainer()
        register_defaults(container, test_settings)

        store: SessionStore = container.resolve(ISessionStore)
        assert store._db_path == test_settings.db_path

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

    def test_register_defaults_includes_subagent_tool(self, test_settings: Settings) -> None:
        container = DIContainer()
        register_defaults(container, test_settings)

        registry: ToolRegistry = container.resolve(IToolRegistry)
        assert "subagent" in registry.tool_names

    async def test_guardrails_block_path_escape(self, test_settings: Settings) -> None:
        """The production pipeline enforces path confinement on real dispatch."""
        container = DIContainer()
        register_defaults(container, test_settings)

        registry: ToolRegistry = container.resolve(IToolRegistry)
        inside = test_settings.work_dir / "f.txt"
        inside.write_text("hi")

        with pytest.raises(ToolError, match="Path traversal"):
            await registry.dispatch("read_file", {"path": "/etc/passwd"})

        result = await registry.dispatch("read_file", {"path": str(inside)})
        assert result == "hi"

    def test_register_defaults_includes_plugin_tools(
        self, test_settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("src.di.factories.discover_tools", lambda: [_EchoTool()])
        container = DIContainer()
        register_defaults(container, test_settings)

        registry: ToolRegistry = container.resolve(IToolRegistry)
        assert "echo" in registry.tool_names

    def test_claude_client_honors_prompt_caching_setting(self, test_settings: Settings) -> None:
        container = DIContainer()
        register_defaults(container, test_settings)

        client: ClaudeClient = container.resolve(LLMClient)
        assert client._prompt_caching == test_settings.prompt_caching_enabled

    def test_prompt_caching_can_be_disabled(self, tmp_path: Path) -> None:
        settings = Settings(
            anthropic_api_key="test-key",
            work_dir=tmp_path,
            memory_dir=tmp_path / "memory",
            db_path=tmp_path / "taskflow.db",
            prompt_caching_enabled=False,
        )
        container = DIContainer()
        register_defaults(container, settings)

        client: ClaudeClient = container.resolve(LLMClient)
        assert client._prompt_caching is False


class TestCreateProductionOrchestrator:
    def test_returns_wired_orchestrator(self, test_settings: Settings) -> None:
        orch = create_production_orchestrator(settings=test_settings)
        assert isinstance(orch, AgentOrchestrator)
        assert orch.tools.tool_names
        assert "subagent" in orch.tools.tool_names
        assert isinstance(orch.memory, ConversationMemory)
        assert orch._max_tool_calls == test_settings.max_tool_calls_per_turn


class TestCreateProductionApp:
    def test_returns_orchestrator_and_session_store(self, test_settings: Settings) -> None:
        app = create_production_app(settings=test_settings)
        assert isinstance(app, AppComponents)
        assert isinstance(app.orchestrator, AgentOrchestrator)
        assert isinstance(app.session_store, SessionStore)
        assert app.session_store._db_path == test_settings.db_path

    def test_returns_task_store(self, test_settings: Settings) -> None:
        app = create_production_app(settings=test_settings)
        assert isinstance(app.task_store, TaskStore)
        assert app.task_store._db_path == test_settings.db_path

    def test_components_share_no_state(self, test_settings: Settings) -> None:
        app = create_production_app(settings=test_settings)
        app.orchestrator.memory.add_user("hi")
        # The session store is independent of conversation memory.
        assert app.session_store.list() == []
