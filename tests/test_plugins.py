"""Tests for plugin discovery and registry integration (network/package-free)."""

from __future__ import annotations

import importlib.metadata
import logging
from types import SimpleNamespace
from typing import Any

import pytest

from src.plugins import discover_tools
from src.tools.base import Tool
from src.tools.registry import ToolRegistry


class _EchoTool(Tool):
    """Real ``Tool`` subclass — doubles as the reference example plugin."""

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


class _DuckTool:
    """Duck-typed Tool-like object (no ``Tool`` base) — honors the documented contract."""

    name = "duck"
    description = "Duck-typed tool"
    input_schema: dict[str, Any] = {"type": "object", "properties": {}}

    async def run(self, **kwargs: Any) -> str:
        return "quack"


class _BrokenTool:
    def __init__(self) -> None:
        raise RuntimeError("boom")


def _echo_factory() -> _EchoTool:
    return _EchoTool()


def _raise_load_error() -> None:
    raise RuntimeError("load boom")


def _raise_type_error(*args: object, **kwargs: object) -> None:
    raise TypeError("No such entry point group")


def _ep(name: str, loaded: Any, module: str = "some_pkg") -> SimpleNamespace:
    return SimpleNamespace(name=name, module=module, load=lambda: loaded)


def _patch_entry_points(monkeypatch: pytest.MonkeyPatch, eps: list[SimpleNamespace]) -> None:
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda **kw: eps)


class TestPluginDiscovery:
    def test_discovers_no_plugins_in_clean_env(self) -> None:
        """Without any installed plugins, discovery returns an empty list."""
        tools = discover_tools()
        assert isinstance(tools, list)
        assert len(tools) == 0

    def test_import_error_falls_back_to_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delattr(importlib.metadata, "entry_points", raising=True)
        assert discover_tools() == []

    def test_type_error_on_unknown_group_returns_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(importlib.metadata, "entry_points", _raise_type_error)
        assert discover_tools() == []

    def test_loads_tool_subclass(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_entry_points(monkeypatch, [_ep("echo", _EchoTool)])
        result = discover_tools()
        assert len(result) == 1
        assert result[0].name == "echo"

    def test_loads_tool_instance_directly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        duck = _DuckTool()
        _patch_entry_points(monkeypatch, [_ep("duck", duck)])
        result = discover_tools()
        assert result == [duck]
        assert result[0] is duck

    def test_calls_factory_callable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_entry_points(monkeypatch, [_ep("echo", _echo_factory)])
        result = discover_tools()
        assert len(result) == 1
        assert result[0].name == "echo"

    def test_skips_incomplete_tool_shape(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        incomplete = SimpleNamespace(name="half", description="missing")
        _patch_entry_points(monkeypatch, [_ep("half", incomplete)])
        with caplog.at_level(logging.WARNING):
            result = discover_tools()
        assert result == []
        assert "does not look like a Tool" in caplog.text

    def test_skips_non_tool_object(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        _patch_entry_points(monkeypatch, [_ep("junk", object())])
        with caplog.at_level(logging.WARNING):
            result = discover_tools()
        assert result == []
        assert "does not look like a Tool" in caplog.text

    def test_skips_entry_point_that_raises_on_load(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        ep = SimpleNamespace(name="broken", module="pkg", load=_raise_load_error)
        _patch_entry_points(monkeypatch, [ep])
        with caplog.at_level(logging.ERROR):
            result = discover_tools()
        assert result == []
        assert "Failed to load plugin broken" in caplog.text

    def test_skips_entry_point_whose_constructor_raises(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        _patch_entry_points(monkeypatch, [_ep("boom", _BrokenTool)])
        with caplog.at_level(logging.ERROR):
            result = discover_tools()
        assert result == []
        assert "Failed to load plugin boom" in caplog.text

    def test_loads_multiple_entry_points_in_order(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_entry_points(monkeypatch, [_ep("echo", _EchoTool), _ep("duck", _DuckTool())])
        result = discover_tools()
        assert [t.name for t in result] == ["echo", "duck"]


class TestRegistryIntegration:
    async def test_registry_registers_plugin_tool(self) -> None:
        registry = ToolRegistry(extra_tools=[_EchoTool()])
        assert "echo" in registry.tool_names
        assert await registry.dispatch("echo", {"text": "hi"}) == "Echo: hi"

    def test_registry_builtin_wins_on_duplicate_name(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        dup = SimpleNamespace(
            name="read_file",
            description="dup",
            input_schema={},
            run=lambda **kw: "x",
        )
        with caplog.at_level(logging.WARNING):
            registry = ToolRegistry(extra_tools=[dup])
        assert registry.tool_names.count("read_file") == 1
        assert "Skipping extra tool" in caplog.text

    def test_registry_first_plugin_wins_on_plugin_duplicate(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        first = _EchoTool()
        second = _EchoTool()
        with caplog.at_level(logging.WARNING):
            registry = ToolRegistry(extra_tools=[first, second])
        assert registry._tools["echo"] is first
        assert "Skipping extra tool" in caplog.text

    async def test_discovered_plugins_feed_registry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_entry_points(monkeypatch, [_ep("echo", _EchoTool)])
        registry = ToolRegistry(extra_tools=discover_tools())
        assert "echo" in registry.tool_names
        assert await registry.dispatch("echo", {"text": "hi"}) == "Echo: hi"
