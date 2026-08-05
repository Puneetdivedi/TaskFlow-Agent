"""Tests for YAML read/write tools."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.tools.base import ToolError
from src.tools.registry import ToolRegistry
from src.tools.yaml_tools import YamlReadTool, YamlWriteTool


@pytest.fixture
def temp_dir() -> Path:
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


# ---------------------------------------------------------------------------
# YamlReadTool
# ---------------------------------------------------------------------------
class TestYamlReadToolContract:
    def test_name(self) -> None:
        assert YamlReadTool().name == "yaml_read"

    def test_schema_requires_path(self) -> None:
        schema = YamlReadTool().input_schema
        assert schema["type"] == "object"
        assert schema["required"] == ["path"]
        assert "path" in schema["properties"]


class TestYamlReadTool:
    async def test_reads_mapping(self, temp_dir: Path) -> None:
        f = temp_dir / "config.yaml"
        f.write_text("name: TaskFlow\nversion: 1")
        result = await YamlReadTool().run(str(f))
        assert result == "name: TaskFlow\nversion: 1"

    async def test_reads_nested_mapping(self, temp_dir: Path) -> None:
        f = temp_dir / "config.yaml"
        f.write_text("server:\n  host: localhost\n  ports:\n    - 80\n    - 443")
        result = await YamlReadTool().run(str(f))
        assert result == "server:\n  host: localhost\n  ports:\n  - 80\n  - 443"

    async def test_reads_sequence(self, temp_dir: Path) -> None:
        f = temp_dir / "list.yaml"
        f.write_text("- alpha\n- beta")
        result = await YamlReadTool().run(str(f))
        assert result == "- alpha\n- beta"

    async def test_reads_scalar_document(self, temp_dir: Path) -> None:
        f = temp_dir / "scalar.yaml"
        f.write_text("hello")
        result = await YamlReadTool().run(str(f))
        assert result == "hello\n..."

    async def test_normalizes_dates_and_types(self, temp_dir: Path) -> None:
        f = temp_dir / "types.yaml"
        f.write_text('date: 2026-08-01\nflag: "false"\ncount: 5')
        result = await YamlReadTool().run(str(f))
        assert result == "date: 2026-08-01\nflag: 'false'\ncount: 5"

    async def test_empty_file(self, temp_dir: Path) -> None:
        f = temp_dir / "empty.yaml"
        f.write_text("")
        assert await YamlReadTool().run(str(f)) == "(empty YAML file)"

    async def test_comment_only_file(self, temp_dir: Path) -> None:
        f = temp_dir / "notes.yaml"
        f.write_text("# nothing here")
        assert await YamlReadTool().run(str(f)) == "(empty YAML file)"

    async def test_file_not_found(self) -> None:
        with pytest.raises(ToolError, match="File not found"):
            await YamlReadTool().run("/nonexistent/config.yaml")

    async def test_not_a_file(self, temp_dir: Path) -> None:
        with pytest.raises(ToolError, match="Not a file"):
            await YamlReadTool().run(str(temp_dir))

    async def test_invalid_yaml(self, temp_dir: Path) -> None:
        f = temp_dir / "bad.yaml"
        f.write_text("a: 1\n  b: 2")
        with pytest.raises(ToolError, match="Invalid YAML"):
            await YamlReadTool().run(str(f))

    async def test_read_failure_is_wrapped(
        self, temp_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        f = temp_dir / "config.yaml"
        f.write_text("a: 1")

        def boom(*args: object, **kwargs: object) -> str:
            raise OSError("permission denied")

        monkeypatch.setattr(Path, "read_text", boom)
        with pytest.raises(ToolError, match="Failed to read"):
            await YamlReadTool().run(str(f))


# ---------------------------------------------------------------------------
# YamlWriteTool
# ---------------------------------------------------------------------------
class TestYamlWriteToolContract:
    def test_name(self) -> None:
        assert YamlWriteTool().name == "yaml_write"

    def test_schema_requires_path_and_content(self) -> None:
        schema = YamlWriteTool().input_schema
        assert schema["type"] == "object"
        assert schema["required"] == ["path", "content"]
        assert "path" in schema["properties"]
        assert "content" in schema["properties"]


class TestYamlWriteTool:
    async def test_writes_normalized_content(self, temp_dir: Path) -> None:
        target = temp_dir / "config.yaml"
        result = await YamlWriteTool().run(str(target), "name: TaskFlow\nversion: 1")
        normalized = "name: TaskFlow\nversion: 1\n"
        assert target.read_text() == normalized
        assert result == f"Wrote {len(normalized)} bytes to {target}"

    async def test_normalizes_flow_style_to_block(self, temp_dir: Path) -> None:
        target = temp_dir / "config.yaml"
        await YamlWriteTool().run(str(target), "{a: 1, b: [2, 3]}")
        assert target.read_text() == "a: 1\nb:\n- 2\n- 3\n"

    async def test_preserves_key_order(self, temp_dir: Path) -> None:
        target = temp_dir / "config.yaml"
        await YamlWriteTool().run(str(target), "zebra: 1\nalpha: 2\nmike: 3")
        assert target.read_text() == "zebra: 1\nalpha: 2\nmike: 3\n"

    async def test_creates_parent_dirs(self, temp_dir: Path) -> None:
        target = temp_dir / "a" / "b" / "c.yaml"
        await YamlWriteTool().run(str(target), "a: 1")
        assert target.exists()
        assert target.read_text() == "a: 1\n"

    async def test_rejects_invalid_yaml_content(self, temp_dir: Path) -> None:
        with pytest.raises(ToolError, match="Invalid YAML content"):
            await YamlWriteTool().run(str(temp_dir / "bad.yaml"), "a: 1\n  b: 2")

    async def test_rejects_empty_content(self, temp_dir: Path) -> None:
        with pytest.raises(ToolError, match="must not be empty"):
            await YamlWriteTool().run(str(temp_dir / "e.yaml"), "")

    async def test_rejects_comment_only_content(self, temp_dir: Path) -> None:
        with pytest.raises(ToolError, match="must not be empty"):
            await YamlWriteTool().run(str(temp_dir / "e.yaml"), "# nothing")

    async def test_rejects_null_document_content(self, temp_dir: Path) -> None:
        with pytest.raises(ToolError, match="must not be empty"):
            await YamlWriteTool().run(str(temp_dir / "e.yaml"), "null")

    async def test_write_failure_is_wrapped(
        self, temp_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = temp_dir / "config.yaml"

        def boom(*args: object, **kwargs: object) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(Path, "write_text", boom)
        with pytest.raises(ToolError, match="Failed to write"):
            await YamlWriteTool().run(str(target), "a: 1")


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------
class TestYamlToolsRegistryIntegration:
    def test_registry_includes_yaml_tools(self) -> None:
        registry = ToolRegistry()
        assert "yaml_read" in registry.tool_names
        assert "yaml_write" in registry.tool_names

    def test_anthropic_defs_include_yaml_tools(self) -> None:
        defs = ToolRegistry().anthropic_tool_defs()
        by_name = {d["name"]: d for d in defs}
        for name in ("yaml_read", "yaml_write"):
            schema = by_name[name]["input_schema"]
            assert schema["type"] == "object"
            assert "properties" in schema

    async def test_dispatch_write_then_read(self, temp_dir: Path) -> None:
        registry = ToolRegistry()
        target = temp_dir / "cfg.yaml"
        write_result = await registry.dispatch(
            "yaml_write", {"path": str(target), "content": "a: 1\n"}
        )
        assert "Wrote 5 bytes" in write_result
        assert target.read_text() == "a: 1\n"
        read_result = await registry.dispatch("yaml_read", {"path": str(target)})
        assert read_result == "a: 1"

    async def test_dispatch_propagates_yaml_errors(self, temp_dir: Path) -> None:
        registry = ToolRegistry()
        with pytest.raises(ToolError, match="Invalid YAML content"):
            await registry.dispatch(
                "yaml_write", {"path": str(temp_dir / "bad.yaml"), "content": "a: [1"}
            )
