"""MCP client support — lets the agent call tools from external MCP servers.

Configure external servers with the ``MCP_SERVERS`` environment variable, a
JSON object mapping a server name to its launch details::

    MCP_SERVERS={"filesystem": {"command": "npx", "args": ["-y",
        "@modelcontextprotocol/server-filesystem", "/tmp"]}}

At startup each configured server is queried for its tool list and every
remote tool is wrapped as a first-class :class:`~src.tools.base.Tool` with
the server-provided name, description, and input schema. When the agent
calls one, the arguments are forwarded to the remote server over stdio and
the returned text becomes the tool result.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Coroutine, TypeVar, cast

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from mcp import types as mcp_types
from src.tools.base import Tool, ToolError

logger = logging.getLogger(__name__)

# How long to wait for a single external server to handshake during startup
# discovery before giving up and skipping it. Sized generously: a cold
# TaskFlow server takes ~8-14s just to import and build its orchestrator
# before it starts serving, so a tight timeout would spuriously skip real
# servers. The bound still stops a hung server from blocking startup.
DISCOVERY_TIMEOUT_S = 30.0

T = TypeVar("T")


@dataclass(frozen=True)
class MCPServerConfig:
    """Connection details for one external MCP server (stdio transport)."""

    name: str
    command: str
    args: tuple[str, ...] = ()
    env: dict[str, str] | None = None
    cwd: Path | None = None


def load_mcp_servers() -> list[MCPServerConfig]:
    """Read configured servers from the ``MCP_SERVERS`` environment variable.

    Expects a JSON object mapping server name → ``{"command": str,
    "args": list[str]?, "env": dict?, "cwd": str?}``. Malformed input is
    logged and skipped — a bad config never crashes startup.
    """
    raw = os.getenv("MCP_SERVERS")
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("MCP_SERVERS is not valid JSON — ignoring MCP client servers")
        return []
    if not isinstance(data, dict):
        logger.warning("MCP_SERVERS must be a JSON object — ignoring MCP client servers")
        return []

    servers: list[MCPServerConfig] = []
    for name, cfg in data.items():
        if not isinstance(cfg, dict) or not isinstance(cfg.get("command"), str):
            logger.warning("Skipping invalid MCP server config %r (missing command)", name)
            continue
        servers.append(
            MCPServerConfig(
                name=name,
                command=cfg["command"],
                args=tuple(cfg.get("args") or ()),
                env=cfg.get("env"),
                cwd=Path(cfg["cwd"]) if cfg.get("cwd") else None,
            )
        )
    return servers


class MCPClientTool(Tool):
    """A TaskFlow tool backed by a named tool on an external MCP server."""

    def __init__(self, server: MCPServerConfig, tool_def: mcp_types.Tool) -> None:
        self._server = server
        self._def = tool_def

    @property
    def name(self) -> str:
        return self._def.name

    @property
    def description(self) -> str:
        return self._def.description or ""

    @property
    def input_schema(self) -> dict[str, Any]:
        return self._def.input_schema

    async def run(self, **kwargs: Any) -> str:
        """Forward *kwargs* to the remote tool and return its text result."""
        params = StdioServerParameters(
            command=self._server.command,
            args=list(self._server.args),
            env=self._server.env,
            cwd=self._server.cwd,
        )
        label = f"MCP server '{self._server.name}' tool '{self._def.name}'"
        try:
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(self._def.name, dict(kwargs))
        except Exception as exc:
            raise ToolError(f"{label}: {exc}") from exc
        return _format_result(result, label=label)


def _format_result(result: mcp_types.CallToolResult, *, label: str) -> str:
    """Turn an MCP ``CallToolResult`` into the text the LLM sees."""
    parts: list[str] = []
    for block in result.content:
        if isinstance(block, mcp_types.TextContent) and block.text:
            parts.append(block.text)
    text = "\n".join(parts).strip()
    if result.structured_content is not None and not text:
        text = json.dumps(result.structured_content)
    if result.is_error:
        raise ToolError(f"{label}: {text or 'tool returned an error'}")
    return text or f"(no text returned from {label})"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
async def _fetch_tools(server: MCPServerConfig) -> list[mcp_types.Tool]:
    """Connect once and return the remote server's advertised tools."""
    params = StdioServerParameters(
        command=server.command,
        args=list(server.args),
        env=server.env,
        cwd=server.cwd,
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            return list(result.tools)


async def _collect_tools(
    servers: list[MCPServerConfig],
) -> list[tuple[MCPServerConfig, list[mcp_types.Tool]]]:
    """Fetch each server's tools, skipping any server that fails."""
    pairs: list[tuple[MCPServerConfig, list[mcp_types.Tool]]] = []
    for server in servers:
        try:
            defs = await asyncio.wait_for(_fetch_tools(server), timeout=DISCOVERY_TIMEOUT_S)
        except Exception as exc:
            logger.warning("MCP client: skipping server '%s': %r", server.name, exc)
            continue
        logger.info("MCP client: discovered %d tool(s) from '%s'", len(defs), server.name)
        pairs.append((server, defs))
    return pairs


def _run_loop_in_worker_thread(coro: Coroutine[Any, Any, T]) -> T:
    """Run *coro* on a fresh event loop inside a worker thread.

    Needed because startup discovery can be triggered from inside an already
    running event loop (the agent REPL), where calling ``asyncio.run``
    directly would fail.
    """
    result: Any = None
    errors: list[BaseException] = []

    def _runner() -> None:
        nonlocal result
        try:
            result = asyncio.run(coro)
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    if errors:
        raise errors[0]
    return cast(T, result)


def build_mcp_tools(servers: list[MCPServerConfig]) -> list[Tool]:
    """Synchronously discover and wrap every configured server's tools.

    A server that fails to connect is logged and skipped — discovery never
    blocks startup beyond the per-server timeout.
    """
    try:
        asyncio.get_running_loop()
        inside_loop = True
    except RuntimeError:
        inside_loop = False

    if inside_loop:
        # Called from inside the agent loop where ``asyncio.run`` is illegal —
        # run discovery on a fresh loop in a worker thread instead.
        pairs = _run_loop_in_worker_thread(_collect_tools(servers))
    else:
        pairs = asyncio.run(_collect_tools(servers))

    tools: list[Tool] = []
    for server, defs in pairs:
        tools.extend(MCPClientTool(server, d) for d in defs)
    return tools
