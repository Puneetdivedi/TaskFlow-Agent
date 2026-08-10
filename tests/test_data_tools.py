"""Tests for the structured-data tools (JSON + CSV)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.tools.base import ToolError
from src.tools.data_tools import (
    CsvAggregateTool,
    CsvReadTool,
    JsonReadTool,
    JsonWriteTool,
)
from src.tools.registry import ToolRegistry

# Every day-to-day tool added this phase, checked once against the registry.
NEW_TOOL_NAMES = [
    "base64_encode",
    "base64_decode",
    "url_encode",
    "url_decode",
    "uuid4",
    "hash_text",
    "word_count",
    "json_read",
    "json_write",
    "csv_read",
    "csv_aggregate",
    "current_date",
    "date_add",
    "days_between",
    "calculator",
    "clipboard_read",
    "clipboard_write",
    "system_info",
]


class TestJsonReadTool:
    def test_name(self) -> None:
        assert JsonReadTool().name == "json_read"

    def test_schema_requires_path(self) -> None:
        schema = JsonReadTool().input_schema
        assert schema["required"] == ["path"]

    async def test_reads_and_normalizes(self, tmp_path: Path) -> None:
        f = tmp_path / "data.json"
        f.write_text('{"b": 1, "a": [1, 2, 3]}')
        result = await JsonReadTool().run(str(f))
        assert json.loads(result) == {"b": 1, "a": [1, 2, 3]}
        assert "\n  " in result  # pretty-printed

    async def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ToolError, match="File not found"):
            await JsonReadTool().run(str(tmp_path / "nope.json"))

    async def test_invalid_json_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.json"
        f.write_text("{not json")
        with pytest.raises(ToolError, match="Invalid JSON"):
            await JsonReadTool().run(str(f))

    async def test_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.json"
        f.write_text("")
        assert await JsonReadTool().run(str(f)) == "(empty JSON file)"


class TestJsonWriteTool:
    def test_name(self) -> None:
        assert JsonWriteTool().name == "json_write"

    async def test_writes_normalized_and_creates_parents(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "out.json"
        result = await JsonWriteTool().run(str(target), '{"a":1,"b":2}')
        assert target.exists()
        assert json.loads(target.read_text()) == {"a": 1, "b": 2}
        assert "Wrote " in result

    async def test_roundtrip(self, tmp_path: Path) -> None:
        target = tmp_path / "roundtrip.json"
        await JsonWriteTool().run(str(target), '[1, 2, {"x": true}]')
        assert await JsonReadTool().run(str(target)) == json.dumps(
            [1, 2, {"x": True}], indent=2, ensure_ascii=False
        )

    async def test_invalid_content_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ToolError, match="Invalid JSON content"):
            await JsonWriteTool().run(str(tmp_path / "x.json"), "{broken")


class TestCsvReadTool:
    def test_name(self) -> None:
        assert CsvReadTool().name == "csv_read"

    async def test_reads_header_rows(self, tmp_path: Path) -> None:
        f = tmp_path / "data.csv"
        f.write_text("name,age\nalice,30\nbob,25\n")
        result = json.loads(await CsvReadTool().run(str(f)))
        assert result == [
            {"name": "alice", "age": "30"},
            {"name": "bob", "age": "25"},
        ]

    async def test_headerless_returns_row_lists(self, tmp_path: Path) -> None:
        f = tmp_path / "raw.csv"
        f.write_text("1,2\n3,4\n")
        result = json.loads(await CsvReadTool().run(str(f), has_header=False))
        assert result == [{"col": ["1", "2"]}, {"col": ["3", "4"]}]

    async def test_custom_delimiter(self, tmp_path: Path) -> None:
        f = tmp_path / "semi.csv"
        f.write_text("a;b\n1;2\n")
        result = json.loads(await CsvReadTool().run(str(f), delimiter=";"))
        assert result == [{"a": "1", "b": "2"}]

    async def test_limit_caps_rows(self, tmp_path: Path) -> None:
        f = tmp_path / "many.csv"
        f.write_text("n\n1\n2\n3\n4\n5\n")
        result = json.loads(await CsvReadTool().run(str(f), limit=2))
        assert result == [{"n": "1"}, {"n": "2"}]

    async def test_where_filter(self, tmp_path: Path) -> None:
        f = tmp_path / "filter.csv"
        f.write_text("name,age\nalice,30\nbob,25\nalice,40\n")
        result = json.loads(
            await CsvReadTool().run(str(f), where_column="name", where_value="alice")
        )
        assert result == [{"name": "alice", "age": "30"}, {"name": "alice", "age": "40"}]

    async def test_where_filter_headerless_by_index(self, tmp_path: Path) -> None:
        f = tmp_path / "raw.csv"
        f.write_text("1,2\n3,4\n")
        result = json.loads(
            await CsvReadTool().run(str(f), has_header=False, where_column="0", where_value="3")
        )
        assert result == [{"col": ["3", "4"]}]

    async def test_where_requires_both_columns(self, tmp_path: Path) -> None:
        f = tmp_path / "data.csv"
        f.write_text("a\n1\n")
        with pytest.raises(ToolError, match="provided together"):
            await CsvReadTool().run(str(f), where_column="a")

    async def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ToolError, match="File not found"):
            await CsvReadTool().run(str(tmp_path / "nope.csv"))


class TestCsvAggregateTool:
    def test_name(self) -> None:
        assert CsvAggregateTool().name == "csv_aggregate"

    async def test_count(self, tmp_path: Path) -> None:
        f = tmp_path / "agg.csv"
        f.write_text("name,value\na,1\nb,2\nc,3\nd,x\n")
        assert await CsvAggregateTool().run(str(f), "value") == "count(value) = 4"

    async def test_sum_avg_min_max_skip_non_numeric(self, tmp_path: Path) -> None:
        f = tmp_path / "agg.csv"
        f.write_text("name,value\na,1\nb,2\nc,3\nd,x\n")
        tool = CsvAggregateTool()
        assert await tool.run(str(f), "value", operation="sum") == "sum(value) = 6"
        assert await tool.run(str(f), "value", operation="avg") == "avg(value) = 2"
        assert await tool.run(str(f), "value", operation="min") == "min(value) = 1"
        assert await tool.run(str(f), "value", operation="max") == "max(value) = 3"

    async def test_float_average_formatted(self, tmp_path: Path) -> None:
        f = tmp_path / "floats.csv"
        f.write_text("n\n1\n2\n")
        assert await CsvAggregateTool().run(str(f), "n", operation="avg") == "avg(n) = 1.5"

    async def test_all_non_numeric_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "text.csv"
        f.write_text("v\nfoo\nbar\n")
        with pytest.raises(ToolError, match="No numeric values"):
            await CsvAggregateTool().run(str(f), "v", operation="sum")

    async def test_missing_column_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "agg.csv"
        f.write_text("name,value\na,1\n")
        with pytest.raises(ToolError, match="Column not found"):
            await CsvAggregateTool().run(str(f), "missing")

    async def test_unknown_operation_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "agg.csv"
        f.write_text("v\n1\n")
        with pytest.raises(ToolError, match="Unsupported operation"):
            await CsvAggregateTool().run(str(f), "v", operation="median")


class TestRegistryWiring:
    def test_day_to_day_tools_registered(self) -> None:
        registry = ToolRegistry()
        for name in NEW_TOOL_NAMES:
            assert name in registry.tool_names, f"missing {name}"
