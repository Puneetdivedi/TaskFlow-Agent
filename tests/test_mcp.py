"""Tests for the TaskFlow MCP server."""

from __future__ import annotations

from pathlib import Path

from mcp import types as mcp_types
from src.mcp.server import SERVER_NAME, TaskFlowMCPServer
from src.tools.registry import ToolRegistry


def _server(registry: ToolRegistry | None = None) -> TaskFlowMCPServer:
    return TaskFlowMCPServer(registry or ToolRegistry())


class TestListTools:
    async def test_advertises_all_registry_tools(self) -> None:
        result = await _server()._list_tools(None, None)
        names = {t.name for t in result.tools}
        # File tools are always registered in a plain ToolRegistry.
        assert "read_file" in names
        assert "write_file" in names
        assert "run_shell" in names
        assert "web_search" in names
        assert "yaml_read" in names

    async def test_passes_through_tool_schema(self) -> None:
        result = await _server()._list_tools(None, None)
        read = next(t for t in result.tools if t.name == "read_file")
        assert read.description == "Read the full contents of a text file at the given path."
        assert read.input_schema == {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative path to the file",
                }
            },
            "required": ["path"],
        }


class TestCallTool:
    async def test_call_tool_reads_file(self, tmp_path: Path) -> None:
        target = tmp_path / "hello.txt"
        target.write_text("hello world", encoding="utf-8")
        server = _server()
        params = mcp_types.CallToolRequestParams(name="read_file", arguments={"path": str(target)})
        result = await server._call_tool(None, params)
        assert result.is_error is False
        assert result.content[0].text == "hello world"  # type: ignore[union-attr]

    async def test_call_unknown_tool_returns_error(self) -> None:
        params = mcp_types.CallToolRequestParams(name="no_such_tool", arguments={})
        result = await _server()._call_tool(None, params)
        assert result.is_error is True
        assert "no_such_tool" in result.content[0].text  # type: ignore[union-attr]

    async def test_call_tool_error_does_not_raise(self, tmp_path: Path) -> None:
        server = _server()
        params = mcp_types.CallToolRequestParams(
            name="read_file", arguments={"path": str(tmp_path / "missing.txt")}
        )
        result = await server._call_tool(None, params)
        assert result.is_error is True
        assert "not found" in result.content[0].text.lower()  # type: ignore[union-attr]

    async def test_call_without_arguments(self) -> None:
        server = _server()
        params = mcp_types.CallToolRequestParams(name="read_file", arguments=None)
        result = await server._call_tool(None, params)
        # No path supplied → the tool raises ToolError, surfaced as an error result.
        assert result.is_error is True


class TestServerMetadata:
    def test_server_name(self) -> None:
        assert SERVER_NAME == "taskflow"


class TestProductionWiring:
    async def test_server_accepts_production_orchestrator_tools(self, tmp_path: Path) -> None:
        from config.settings import Settings
        from src.di.factories import create_production_orchestrator

        settings = Settings(
            anthropic_api_key="test-key",
            work_dir=tmp_path,
            memory_dir=tmp_path / "memory",
            db_path=tmp_path / "taskflow.db",
        )
        orch = create_production_orchestrator(settings=settings)
        server = TaskFlowMCPServer(orch.tools)
        result = await server._list_tools(None, None)
        names = {t.name for t in result.tools}
        # A wired registry includes the task tool backed by the temp db.
        assert "tasks" in names
        assert "read_file" in names
