"""Model Context Protocol server exposing TaskFlow's tools.

The server advertises and dispatches every tool from a
:class:`~src.interfaces.tool_registry.IToolRegistry` exactly as the
interactive agent sees it — same names, descriptions, and input schemas —
so any MCP client (Claude Desktop, Claude Code, or a third-party host) can
drive TaskFlow's file, shell, web, YAML, and task tools over stdio.

Run with::

    python -m src.mcp
"""

from __future__ import annotations

import logging
from typing import Any

from mcp.server.lowlevel import Server

from mcp import types as mcp_types
from mcp.server import InitializationOptions, ServerRequestContext, stdio
from src.interfaces.tool_registry import IToolRegistry

logger = logging.getLogger(__name__)

SERVER_NAME = "taskflow"
SERVER_VERSION = "0.1.0"


class TaskFlowMCPServer:
    """Advertise and dispatch a tool registry through the Model Context Protocol."""

    def __init__(self, registry: IToolRegistry) -> None:
        self._registry = registry
        self._server = Server(
            SERVER_NAME,
            on_list_tools=self._list_tools,
            on_call_tool=self._call_tool,
        )

    # ------------------------------------------------------------------
    async def _list_tools(
        self,
        ctx: ServerRequestContext[Any],
        params: mcp_types.PaginatedRequestParams | None = None,
    ) -> mcp_types.ListToolsResult:
        del ctx, params
        tools = [
            mcp_types.Tool(
                name=defn["name"],
                description=defn["description"],
                input_schema=defn["input_schema"],
            )
            for defn in self._registry.anthropic_tool_defs()
        ]
        return mcp_types.ListToolsResult(tools=tools)

    # ------------------------------------------------------------------
    async def _call_tool(
        self,
        ctx: ServerRequestContext[Any],
        params: mcp_types.CallToolRequestParams,
    ) -> mcp_types.CallToolResult:
        del ctx
        name = params.name
        arguments = dict(params.arguments or {})
        logger.info("MCP tool call: %s", name)
        try:
            text = await self._registry.dispatch(name, arguments)
        except Exception as exc:  # noqa: BLE001 - tool errors become MCP error results
            logger.exception("MCP tool %s failed", name)
            return mcp_types.CallToolResult(
                content=[mcp_types.TextContent(type="text", text=f"{name}: {exc}")],
                is_error=True,
            )
        return mcp_types.CallToolResult(
            content=[mcp_types.TextContent(type="text", text=text)],
            is_error=False,
        )

    # ------------------------------------------------------------------
    async def run_stdio(self) -> None:
        """Serve tools over stdin/stdout until the client disconnects."""
        async with stdio.stdio_server() as (read_stream, write_stream):
            await self._server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name=SERVER_NAME,
                    server_version=SERVER_VERSION,
                    capabilities=self._server.get_capabilities(),
                ),
            )
