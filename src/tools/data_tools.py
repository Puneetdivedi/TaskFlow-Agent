"""Structured-data tools — JSON read/write and CSV read/aggregate."""

from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
from pathlib import Path
from typing import Any

from src.tools.base import Tool, ToolError

logger = logging.getLogger(__name__)

#: CSV operations supported by :class:`CsvAggregateTool`.
CSV_OPERATIONS: tuple[str, ...] = ("count", "sum", "avg", "min", "max")


def _resolve_path(path: str) -> Path:
    return Path(path).expanduser().resolve()


def _format_number(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:g}"


class JsonReadTool(Tool):
    """Read a JSON file and return its parsed contents as normalized JSON."""

    @property
    def name(self) -> str:
        return "json_read"

    @property
    def description(self) -> str:
        return "Read a JSON file and return its contents as normalized, pretty-printed JSON."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative path to the JSON file",
                }
            },
            "required": ["path"],
        }

    async def run(self, path: str, **kwargs: Any) -> str:  # type: ignore[override]
        p = _resolve_path(path)
        if not p.exists():
            raise ToolError(f"File not found: {p}")
        if not p.is_file():
            raise ToolError(f"Not a file: {p}")
        logger.debug("Reading JSON file: %s", p)
        try:
            text = await asyncio.to_thread(p.read_text, encoding="utf-8")
        except Exception as exc:
            raise ToolError(f"Failed to read {p}: {exc}") from exc
        if not text.strip():
            return "(empty JSON file)"
        try:
            parsed: object = json.loads(text)
        except (json.JSONDecodeError, RecursionError) as exc:
            raise ToolError(f"Invalid JSON in {p}: {exc}") from exc
        normalized: str = json.dumps(parsed, indent=2, ensure_ascii=False)
        return normalized


class JsonWriteTool(Tool):
    """Write structured JSON to a file (creates parent directories)."""

    @property
    def name(self) -> str:
        return "json_write"

    @property
    def description(self) -> str:
        return "Write structured JSON to a file (parses and normalizes the content)."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to write to"},
                "content": {"type": "string", "description": "JSON text to write"},
            },
            "required": ["path", "content"],
        }

    async def run(self, path: str, content: str, **kwargs: Any) -> str:  # type: ignore[override]
        p = _resolve_path(path)
        try:
            parsed: object = json.loads(content)
        except (json.JSONDecodeError, RecursionError) as exc:
            raise ToolError(f"Invalid JSON content: {exc}") from exc
        normalized: str = json.dumps(parsed, indent=2, ensure_ascii=False)
        logger.debug("Writing JSON file: %s (%d bytes)", p, len(normalized))
        await asyncio.to_thread(p.parent.mkdir, parents=True, exist_ok=True)
        try:
            await asyncio.to_thread(p.write_text, normalized, encoding="utf-8")
        except Exception as exc:
            raise ToolError(f"Failed to write {p}: {exc}") from exc
        return f"Wrote {len(normalized)} bytes to {p}"


class CsvReadTool(Tool):
    """Read a CSV file and return the rows as JSON."""

    @property
    def name(self) -> str:
        return "csv_read"

    @property
    def description(self) -> str:
        return (
            "Read a CSV file and return the rows as JSON. Optionally filter on one "
            "column and cap the number of rows (limit=0 means all)."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the CSV file"},
                "delimiter": {
                    "type": "string",
                    "description": "Field delimiter (default: ,)",
                },
                "has_header": {
                    "type": "boolean",
                    "description": "Treat the first row as a header (default: true)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max rows to return (default: 100, 0 = all)",
                },
                "where_column": {
                    "type": "string",
                    "description": "Column name (or 0-based index without a header) to filter on",
                },
                "where_value": {
                    "type": "string",
                    "description": "Exact value rows must match in where_column",
                },
            },
            "required": ["path"],
        }

    async def run(self, path: str, **kwargs: Any) -> str:  # type: ignore[override]
        p = _resolve_path(path)
        if not p.exists():
            raise ToolError(f"File not found: {p}")
        if not p.is_file():
            raise ToolError(f"Not a file: {p}")
        delimiter: str = kwargs.get("delimiter", ",")
        if not isinstance(delimiter, str) or len(delimiter) != 1:
            raise ToolError("delimiter must be a single character")
        has_header = bool(kwargs.get("has_header", True))
        limit = kwargs.get("limit", 100)
        where_column = kwargs.get("where_column")
        where_value = kwargs.get("where_value")
        if limit is not None and not isinstance(limit, int):
            raise ToolError("limit must be an integer")
        if (where_column is None) != (where_value is None):
            raise ToolError("where_column and where_value must be provided together")
        try:
            text = await asyncio.to_thread(p.read_text, encoding="utf-8")
        except Exception as exc:
            raise ToolError(f"Failed to read {p}: {exc}") from exc
        rows: list[Any]
        try:
            if has_header:
                rows = [dict(r) for r in csv.DictReader(io.StringIO(text), delimiter=delimiter)]
            else:
                rows = [{"col": r} for r in csv.reader(io.StringIO(text), delimiter=delimiter)]
        except csv.Error as exc:
            raise ToolError(f"Failed to parse CSV {p}: {exc}") from exc

        if where_column is not None:
            column = str(where_column)
            value = str(where_value)
            if has_header:
                rows = [r for r in rows if r.get(column) == value]
            else:
                try:
                    index = int(column)
                except ValueError as exc:
                    raise ToolError(
                        "where_column must be a 0-based index when has_header=false"
                    ) from exc
                rows = [r for r in rows if index < len(r["col"]) and r["col"][index] == value]

        if limit:
            rows = rows[:limit]
        return json.dumps(rows, ensure_ascii=False, indent=2)


class CsvAggregateTool(Tool):
    """Compute a simple aggregate over one column of a CSV file."""

    @property
    def name(self) -> str:
        return "csv_aggregate"

    @property
    def description(self) -> str:
        return (
            "Compute count, sum, avg, min, or max over one column of a CSV file. "
            "Non-numeric cells are skipped for numeric operations."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the CSV file"},
                "column": {
                    "type": "string",
                    "description": "Column name (or 0-based index without a header) to aggregate",
                },
                "operation": {
                    "type": "string",
                    "description": "Aggregate to compute (default: count)",
                    "enum": list(CSV_OPERATIONS),
                },
                "delimiter": {
                    "type": "string",
                    "description": "Field delimiter (default: ,)",
                },
                "has_header": {
                    "type": "boolean",
                    "description": "Treat the first row as a header (default: true)",
                },
            },
            "required": ["path", "column"],
        }

    async def run(self, path: str, column: str, **kwargs: Any) -> str:  # type: ignore[override]
        p = _resolve_path(path)
        if not p.exists():
            raise ToolError(f"File not found: {p}")
        if not p.is_file():
            raise ToolError(f"Not a file: {p}")
        operation: str = kwargs.get("operation", "count")
        if operation not in CSV_OPERATIONS:
            raise ToolError(
                f"Unsupported operation: {operation!r} (choose from {', '.join(CSV_OPERATIONS)})"
            )
        delimiter: str = kwargs.get("delimiter", ",")
        if not isinstance(delimiter, str) or len(delimiter) != 1:
            raise ToolError("delimiter must be a single character")
        has_header = bool(kwargs.get("has_header", True))
        try:
            text = await asyncio.to_thread(p.read_text, encoding="utf-8")
        except Exception as exc:
            raise ToolError(f"Failed to read {p}: {exc}") from exc
        values: list[Any]
        try:
            if has_header:
                records = [dict(r) for r in csv.DictReader(io.StringIO(text), delimiter=delimiter)]
                values = [r.get(column) for r in records]
            else:
                raw = [list(r) for r in csv.reader(io.StringIO(text), delimiter=delimiter)]
                try:
                    index = int(column)
                except ValueError as exc:
                    raise ToolError("column must be a 0-based index when has_header=false") from exc
                values = [row[index] if index < len(row) else None for row in raw]
        except csv.Error as exc:
            raise ToolError(f"Failed to parse CSV {p}: {exc}") from exc

        if has_header and any(v is None for v in values):
            raise ToolError(f"Column not found in {p}: {column!r}")
        cells = [v for v in values if v is not None and v != ""]
        if operation == "count":
            return f"count({column}) = {len(cells)}"
        numbers: list[float] = []
        for cell in cells:
            try:
                numbers.append(float(cell))
            except ValueError:
                continue
        if not numbers:
            raise ToolError(f"No numeric values found in column {column!r}")
        if operation == "sum":
            result = sum(numbers)
        elif operation == "avg":
            result = sum(numbers) / len(numbers)
        elif operation == "min":
            result = min(numbers)
        else:
            result = max(numbers)
        return f"{operation}({column}) = {_format_number(result)}"
