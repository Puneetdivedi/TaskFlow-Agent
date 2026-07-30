"""Central registry that holds all available tools and dispatches calls."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
from src.tools.shell_tools import RunShellTool


class ToolRegistry:
    """Holds all tools and provides Anthropic-format descriptors + dispatch."""

    def __init__(
        self,
        work_dir: Path | None = None,
        safety_level: int = 1,
    ) -> None:
        self._tools: dict[str, Tool] = {}

        # --- file tools ---
        self._register(ReadFileTool())
        self._register(WriteFileTool())
        self._register(ListFilesTool())
        self._register(SearchFilesTool())
        self._register(MoveFileTool())
        self._register(DeleteFileTool())
        self._register(FileIndexTool())

        # --- shell ---
        self._register(RunShellTool(work_dir=work_dir, safety_level=safety_level))

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
        """
        tool = self._tools.get(name)
        if tool is None:
            raise ToolError(f"Unknown tool: {name!r} (available: {', '.join(self.tool_names)})")
        try:
            return await tool.run(**arguments)
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(f"Tool {name} failed: {exc}") from exc
