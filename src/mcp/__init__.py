"""MCP server exposing TaskFlow's tools to external Model Context Protocol clients."""

from src.mcp.server import SERVER_NAME, SERVER_VERSION, TaskFlowMCPServer

__all__ = ["SERVER_NAME", "SERVER_VERSION", "TaskFlowMCPServer"]
