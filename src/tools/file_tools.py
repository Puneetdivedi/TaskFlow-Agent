"""File-system tools: read, write, list, search, and organize files."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from src.memory.file_index import FileIndex
from src.tools.base import Tool, ToolError

logger = logging.getLogger(__name__)


class ReadFileTool(Tool):
    """Read the contents of a file."""

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Read the full contents of a text file at the given path."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative path to the file",
                }
            },
            "required": ["path"],
        }

    async def run(self, path: str, **kwargs: Any) -> str:  # type: ignore[override]
        p = Path(path).expanduser().resolve()
        if not p.exists():
            raise ToolError(f"File not found: {p}")
        if not p.is_file():
            raise ToolError(f"Not a file: {p}")
        logger.debug("Reading file: %s", p)
        try:
            content = await asyncio.to_thread(p.read_text, encoding="utf-8")
        except Exception as exc:
            raise ToolError(f"Failed to read {p}: {exc}") from exc
        return content


class WriteFileTool(Tool):
    """Write content to a file (creates parent directories)."""

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "Write text content to a file. Creates parent directories if they don't exist."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to write to"},
                "content": {"type": "string", "description": "Text content to write"},
            },
            "required": ["path", "content"],
        }

    async def run(self, path: str, content: str, **kwargs: Any) -> str:  # type: ignore[override]
        p = Path(path).expanduser().resolve()
        logger.debug("Writing file: %s (%d bytes)", p, len(content))
        await asyncio.to_thread(p.parent.mkdir, parents=True, exist_ok=True)
        try:
            await asyncio.to_thread(p.write_text, content, encoding="utf-8")
        except Exception as exc:
            raise ToolError(f"Failed to write {p}: {exc}") from exc
        return f"Wrote {len(content)} bytes to {p}"


class ListFilesTool(Tool):
    """List files and directories at a path."""

    @property
    def name(self) -> str:
        return "list_files"

    @property
    def description(self) -> str:
        return (
            "List the contents of a directory. Returns names, sizes, and last-modified timestamps."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path (default: working directory)",
                },
                "pattern": {
                    "type": "string",
                    "description": "Optional glob filter, e.g. '*.py' or '**/*.md'",
                },
            },
            "required": [],
        }

    async def run(self, path: str | None = None, pattern: str | None = None, **kwargs: Any) -> str:
        p = Path(path).expanduser().resolve() if path else Path.cwd()
        if not p.is_dir():
            raise ToolError(f"Not a directory: {p}")

        result = await asyncio.to_thread(self._list_files_sync, p, pattern)
        if result is None:
            return f"(empty directory: {p})"
        header, lines = result
        return header + "\n" + "\n".join(lines)

    @staticmethod
    def _list_files_sync(p: Path, pattern: str | None) -> tuple[str, list[str]] | None:
        """Sync helper — runs in a thread to avoid blocking the event loop."""
        if pattern:
            items = list(p.glob(pattern))
        else:
            items = list(p.iterdir())

        if not items:
            return None

        lines: list[str] = []
        for item in sorted(items, key=lambda x: (not x.is_dir(), x.name.lower())):
            suffix = "/" if item.is_dir() else ""
            size = item.stat().st_size if item.is_file() else ""
            mtime = item.stat().st_mtime
            modified = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
            lines.append(f"{modified}  {size:>8,}  {item.name}{suffix}")

        header = f"{p}/  ({len(items)} entries)"
        return header, lines


class SearchFilesTool(Tool):
    """Search for text inside files using ripgrep-style pattern matching."""

    @property
    def name(self) -> str:
        return "search_files"

    @property
    def description(self) -> str:
        return (
            "Search file contents for a regex pattern. Returns matching file paths "
            "and the matching lines with line numbers."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for",
                },
                "path": {
                    "type": "string",
                    "description": "Root directory to search in (default: working directory)",
                },
                "glob": {
                    "type": "string",
                    "description": "File glob filter, e.g. '*.py' or '**/*.md'",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum matches to return (default 50)",
                },
            },
            "required": ["pattern"],
        }

    async def run(  # type: ignore[override]
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        max_results: int = 50,
        **kwargs: Any,
    ) -> str:
        root = Path(path).expanduser().resolve() if path else Path.cwd()
        cmd = ["rg", "--no-heading", "--line-number", "--color", "never"]
        if glob:
            cmd.extend(["--glob", glob])
        cmd.extend([pattern, str(root)])

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            return (
                "ripgrep (rg) not found. Install it or use a fallback search. "
                "https://github.com/BurntSushi/ripgrep"
            )

        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return f"Search timed out (pattern={pattern!r})"

        stdout_str = stdout.decode("utf-8") if stdout else ""
        stderr_str = stderr.decode("utf-8") if stderr else ""

        if process.returncode not in (0, 1):
            return f"Search error: {stderr_str.strip()}"

        lines = stdout_str.strip().splitlines()
        if not lines:
            return "No matches found."

        truncated = len(lines) > max_results
        matches = lines[:max_results]
        summary = f"Found {len(lines)} match(es) in {root}"
        if truncated:
            summary += f" (showing first {max_results})"
        return summary + "\n" + "\n".join(matches)


class MoveFileTool(Tool):
    """Move or rename a file or directory."""

    @property
    def name(self) -> str:
        return "move_file"

    @property
    def description(self) -> str:
        return "Move or rename a file or directory."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Current path"},
                "dest": {"type": "string", "description": "Target path"},
            },
            "required": ["source", "dest"],
        }

    async def run(self, source: str, dest: str, **kwargs: Any) -> str:  # type: ignore[override]
        src = Path(source).expanduser().resolve()
        dst = Path(dest).expanduser().resolve()
        if not src.exists():
            raise ToolError(f"Source not found: {src}")
        logger.warning("Moving file: %s → %s", src, dst)
        await asyncio.to_thread(dst.parent.mkdir, parents=True, exist_ok=True)
        try:
            await asyncio.to_thread(shutil.move, str(src), str(dst))
        except Exception as exc:
            raise ToolError(f"Failed to move {src} → {dst}: {exc}") from exc
        return f"Moved {src.name} → {dst}"


class DeleteFileTool(Tool):
    """Delete a file or empty directory."""

    @property
    def name(self) -> str:
        return "delete_file"

    @property
    def description(self) -> str:
        return "Delete a file or an empty directory. Use with caution."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to delete"},
                "recursive": {
                    "type": "boolean",
                    "description": "Delete directories recursively (default false)",
                },
            },
            "required": ["path"],
        }

    async def run(  # type: ignore[override]
        self,
        path: str,
        recursive: bool = False,
        **kwargs: Any,
    ) -> str:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            raise ToolError(f"Not found: {p}")

        logger.warning("Deleting: %s (recursive=%s)", p, recursive)
        try:
            if p.is_dir():
                if recursive:
                    await asyncio.to_thread(shutil.rmtree, p)
                else:
                    await asyncio.to_thread(os.rmdir, p)
            else:
                await asyncio.to_thread(p.unlink)
        except OSError as exc:
            raise ToolError(f"Failed to delete {p}: {exc}") from exc

        return f"Deleted {p}"


class FileIndexTool(Tool):
    """Index the filesystem so the agent can quickly answer structure questions."""

    def __init__(self) -> None:
        self._index = FileIndex()

    @property
    def name(self) -> str:
        return "file_index"

    @property
    def description(self) -> str:
        return (
            "Manage a cached index of the file system. Use 'refresh' to walk a "
            "directory tree and record all files/dirs with sizes and timestamps. "
            "Use 'query' to see what's already indexed."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["refresh", "query"],
                    "description": "'refresh' walks a directory and builds the index; "
                    "'query' returns a summary of what's cached",
                },
                "path": {
                    "type": "string",
                    "description": (
                        "Directory path to refresh or query (default: working directory)"
                    ),
                },
            },
            "required": ["action"],
        }

    async def run(  # type: ignore[override]
        self,
        action: str,
        path: str | None = None,
        **kwargs: Any,
    ) -> str:
        if action == "refresh":
            root = Path(path).expanduser().resolve() if path else Path.cwd()
            return await asyncio.to_thread(self._index.refresh, root)
        elif action == "query":
            return await asyncio.to_thread(self._index.query, path)
        else:
            raise ToolError(f"Unknown file_index action: {action!r}")
