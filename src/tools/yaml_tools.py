"""YAML tools - read and write structured YAML files."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import yaml

from src.tools.base import Tool, ToolError

logger = logging.getLogger(__name__)


class YamlReadTool(Tool):
    """Read a YAML file and return its parsed contents as canonical YAML."""

    @property
    def name(self) -> str:
        return "yaml_read"

    @property
    def description(self) -> str:
        return "Read a YAML file and return its contents as normalized, canonical YAML."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative path to the YAML file",
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
        logger.debug("Reading YAML file: %s", p)
        try:
            text = await asyncio.to_thread(p.read_text, encoding="utf-8")
        except Exception as exc:
            raise ToolError(f"Failed to read {p}: {exc}") from exc
        try:
            parsed: object = yaml.safe_load(text)
            if parsed is None:
                return "(empty YAML file)"
            normalized: str = yaml.safe_dump(
                parsed, sort_keys=False, allow_unicode=True, default_flow_style=False
            )
            return normalized.rstrip("\n")
        except (yaml.YAMLError, RecursionError) as exc:
            raise ToolError(f"Invalid YAML in {p}: {exc}") from exc


class YamlWriteTool(Tool):
    """Write structured YAML to a file (creates parent directories)."""

    @property
    def name(self) -> str:
        return "yaml_write"

    @property
    def description(self) -> str:
        return "Write structured YAML to a file (parses and normalizes the content)."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to write to"},
                "content": {"type": "string", "description": "YAML text to write"},
            },
            "required": ["path", "content"],
        }

    async def run(self, path: str, content: str, **kwargs: Any) -> str:  # type: ignore[override]
        p = Path(path).expanduser().resolve()
        try:
            parsed: object = yaml.safe_load(content)
            if parsed is None:
                raise ToolError("yaml_write: content must not be empty")
            normalized: str = yaml.safe_dump(
                parsed, sort_keys=False, allow_unicode=True, default_flow_style=False
            )
        except (yaml.YAMLError, RecursionError) as exc:
            raise ToolError(f"Invalid YAML content: {exc}") from exc
        logger.debug("Writing YAML file: %s (%d bytes)", p, len(normalized))
        await asyncio.to_thread(p.parent.mkdir, parents=True, exist_ok=True)
        try:
            await asyncio.to_thread(p.write_text, normalized, encoding="utf-8")
        except Exception as exc:
            raise ToolError(f"Failed to write {p}: {exc}") from exc
        return f"Wrote {len(normalized)} bytes to {p}"
