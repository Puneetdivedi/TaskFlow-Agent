"""Run TaskFlow's tool registry as a Model Context Protocol server over stdio.

Usage::

    python -m src.mcp

Point an MCP client (Claude Desktop, Claude Code, etc.) at this process and
it will see every TaskFlow tool (read_file, run_shell, web_search, tasks, …)
with the same schemas the interactive agent uses.
"""

from __future__ import annotations

import asyncio

from src.di.factories import create_production_orchestrator
from src.logging_config import configure_logging
from src.mcp.server import TaskFlowMCPServer


async def _serve() -> None:
    orchestrator = create_production_orchestrator()
    server = TaskFlowMCPServer(orchestrator.tools)
    await server.run_stdio()


def main() -> None:
    configure_logging()
    asyncio.run(_serve())


if __name__ == "__main__":
    main()
