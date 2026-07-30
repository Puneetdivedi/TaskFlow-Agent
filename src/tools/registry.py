"""Central registry that holds all available tools and dispatches calls."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from src.tools.base import Tool, ToolError
from src.tools.file_tools import (
    DeleteFileTool,
    FileIndexTool,
    ListFilesTool,
    MoveFileTool,
    ReadFileTool,
    SearchFilesTool,
    WriteFileTool,
)
from src.tools.middleware import ToolPipeline
from src.tools.shell_tools import RunShellTool

if False:  # TYPE_CHECKING without runtime circulars
    pass


class ToolRegistry:
    """Holds all tools and provides Anthropic-format descriptors + dispatch.

    Optionally accepts a middleware pipeline and extra third-party tools.
    """

    def __init__(
        self,
        work_dir: Path | None = None,
        safety_level: int = 1,
        extra_tools: list[Tool] | None = None,
        pipeline: ToolPipeline | None = None,
    ) -> None:
        self._tools: dict[str, Tool] = {}
        self._pipeline = pipeline

        # --- Built-in file tools ---
        self._register(ReadFileTool())
        self._register(WriteFileTool())
        self._register(ListFilesTool())
        self._register(SearchFilesTool())
        self._register(MoveFileTool())
        self._register(DeleteFileTool())
        self._register(FileIndexTool())

        # --- Built-in shell tool ---
        self._register(RunShellTool(work_dir=work_dir, safety_level=safety_level))

        # --- Extra (plugin) tools ---
        for t in (extra_tools or []):
            self._register(t)

        logger.info("ToolRegistry initialised with %d tools", len(self._tools))

    # ------------------------------------------------------------------
    def _register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    # ------------------------------------------------------------------
    def anthropic_tool_defs(self) -> list[dict]:
        """Return tool descriptors in Anthropic's tool-use format."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in self._tools.values()
        ]

    # ------------------------------------------------------------------
    async def dispatch(self, name: str, arguments: dict[str, Any]) -> str:
        """Look up a tool by name, call it, and return its string result.

        Raises ToolError if the tool isn't found or execution fails.
        When a middleware pipeline is configured, it wraps the dispatch.
        """
        if self._pipeline is not None:
            return await self._pipeline.run(self._run_tool, name, arguments)

        return await self._run_tool(name, arguments)

    async def _run_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Direct tool execution — no middleware wrapper."""
        tool = self._tools.get(name)
        if tool is None:
            logger.warning("Unknown tool requested: %r", name)
            raise ToolError(f"Unknown tool: {name!r} (available: {', '.join(self.tool_names)})")
        logger.debug("Dispatching tool: %s", name)
        try:
            return await tool.run(**arguments)
        except ToolError:
            raise
        except Exception as exc:
            logger.error("Tool %s raised unexpected error: %s", name, exc)
            raise ToolError(f"Tool {name} failed: {exc}") from exc
