"""Tests for the TaskFlow MCP client (calling tools on external MCP servers)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from mcp import types as mcp_types
from src.mcp.client import (
    MCPClientTool,
    MCPServerConfig,
    _format_result,
    build_mcp_tools,
    load_mcp_servers,
)
from src.tools.base import ToolError


def _tool_def(name: str = "remote_tool", *, description: str = "A remote tool") -> mcp_types.Tool:
    return mcp_types.Tool(
        name=name,
        description=description,
        input_schema={"type": "object", "properties": {}},
    )


def _server(name: str = "fs") -> MCPServerConfig:
    return MCPServerConfig(name=name, command="npx", args=("-y", "some-server"))


def _text_result(text: str, *, is_error: bool = False) -> mcp_types.CallToolResult:
    return mcp_types.CallToolResult(
        content=[mcp_types.TextContent(type="text", text=text)],
        is_error=is_error,
    )


class TestLoadMCPServers:
    def test_unset_env_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MCP_SERVERS", raising=False)
        assert load_mcp_servers() == []

    def test_parses_valid_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(
            "MCP_SERVERS",
            '{"filesystem": {"command": "npx", "args": ["-y", "srv"], "env": {"A": "1"}}}',
        )
        servers = load_mcp_servers()
        assert len(servers) == 1
        assert servers[0].name == "filesystem"
        assert servers[0].command == "npx"
        assert servers[0].args == ("-y", "srv")
        assert servers[0].env == {"A": "1"}
        assert servers[0].cwd is None

    def test_parses_cwd(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MCP_SERVERS", '{"s": {"command": "cmd", "cwd": "/some/dir"}}')
        servers = load_mcp_servers()
        assert servers[0].cwd == Path("/some/dir")

    def test_invalid_json_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MCP_SERVERS", "not json")
        assert load_mcp_servers() == []

    def test_non_object_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MCP_SERVERS", "[1, 2]")
        assert load_mcp_servers() == []

    def test_entry_without_command_is_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MCP_SERVERS", '{"bad": {"args": []}, "good": {"command": "c"}}')
        servers = load_mcp_servers()
        assert [s.name for s in servers] == ["good"]


class TestMCPClientToolProperties:
    def test_passes_through_name_description_schema(self) -> None:
        tool = MCPClientTool(_server(), _tool_def(name="my_tool", description="Does things"))
        assert tool.name == "my_tool"
        assert tool.description == "Does things"
        assert tool.input_schema == {"type": "object", "properties": {}}

    def test_missing_description_defaults_to_empty(self) -> None:
        tool = MCPClientTool(_server(), _tool_def(description=""))
        assert tool.description == ""


class TestFormatResult:
    def test_joins_text_blocks(self) -> None:
        result = mcp_types.CallToolResult(
            content=[
                mcp_types.TextContent(type="text", text="line one"),
                mcp_types.TextContent(type="text", text="line two"),
            ],
            is_error=False,
        )
        assert _format_result(result, label="srv t") == "line one\nline two"

    def test_structured_content_fallback(self) -> None:
        result = mcp_types.CallToolResult(
            content=[],
            structured_content=[{"role": "assistant", "content": "hi"}],
            is_error=False,
        )
        assert _format_result(result, label="srv t") == '[{"role": "assistant", "content": "hi"}]'

    def test_error_result_raises_tool_error(self) -> None:
        with pytest.raises(ToolError, match="boom"):
            _format_result(_text_result("boom", is_error=True), label="srv t")

    def test_empty_text_result(self) -> None:
        result = _text_result("")
        assert "no text returned" in _format_result(result, label="srv t")


class TestBuildMCPTools:
    def test_no_servers_yields_no_tools(self) -> None:
        assert build_mcp_tools([]) == []

    async def test_discovers_and_calls_taskflow_itself(self, tmp_path: Path) -> None:
        # Spin up TaskFlow's own MCP server (python -m src.mcp) as the remote
        # endpoint, point the client at it, and round-trip a tool call. Uses a
        # temp db and an empty MCP_SERVERS so the server process never tries to
        # connect to anything itself.
        env = {
            **os.environ,
            "TASKFLOW_DB": str(tmp_path / "db" / "taskflow.db"),
            "MCP_SERVERS": "",
        }
        server = MCPServerConfig(
            name="taskflow",
            command=sys.executable,
            args=("-m", "src.mcp"),
            env=env,
        )

        tools = build_mcp_tools([server])
        by_name = {t.name: t for t in tools}
        assert "read_file" in by_name
        assert "tasks" in by_name

        target = tmp_path / "hello.txt"
        target.write_text("hello from mcp client", encoding="utf-8")
        result = await by_name["read_file"].run(path=str(target))
        assert result == "hello from mcp client"

    async def test_bad_server_is_skipped(self) -> None:
        server = MCPServerConfig(
            name="broken",
            command="definitely-not-a-real-command-xyz",
            args=(),
        )
        assert build_mcp_tools([server]) == []
