"""Central registry that holds all available tools and dispatches calls."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.interfaces.task_store import ITaskStore
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
from src.tools.task_tools import TaskTool
from src.tools.web_tools import WebFetchTool, WebSearchTool
from src.tools.yaml_tools import YamlReadTool, YamlWriteTool

logger = logging.getLogger(__name__)

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
        task_store: ITaskStore | None = None,
        file_index_db: Path | None = None,
    ) -> None:
        self._tools: dict[str, Tool] = {}
        self._pipeline = pipeline

        # --- Built-in file tools ---
        self.add_tool(ReadFileTool())
        self.add_tool(WriteFileTool())
        self.add_tool(ListFilesTool())
        self.add_tool(SearchFilesTool())
        self.add_tool(MoveFileTool())
        self.add_tool(DeleteFileTool())
        self.add_tool(FileIndexTool(db_path=file_index_db))

        # --- Built-in shell tool ---
        self.add_tool(RunShellTool(work_dir=work_dir, safety_level=safety_level))

        # --- Built-in web tools (read-only; always available, no safety gating) ---
        self.add_tool(WebSearchTool())
        self.add_tool(WebFetchTool())

        # --- Built-in YAML tools (structured config/spec read/write) ---
        self.add_tool(YamlReadTool())
        self.add_tool(YamlWriteTool())

        # --- Built-in task tool ---
        if task_store is not None:
            self.add_tool(TaskTool(task_store))

        # --- Extra (plugin) tools ---
        for t in extra_tools or []:
            if t.name in self._tools:
                logger.warning("Skipping extra tool %r: name already registered", t.name)
                continue
            self.add_tool(t)

        logger.info("ToolRegistry initialised with %d tools", len(self._tools))

    # ------------------------------------------------------------------
    def add_tool(self, tool: Tool) -> None:
        """Register *tool* by name, raising on a duplicate name."""
        if tool.name in self._tools:
            raise ValueError(f"Duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    # ------------------------------------------------------------------
    def anthropic_tool_defs(self) -> list[dict[str, Any]]:
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
