"""Tests for file-system tools."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from src.tools.base import ToolError
from src.tools.file_tools import (
    DeleteFileTool,
    ListFilesTool,
    ReadFileTool,
    SearchFilesTool,
    WriteFileTool,
)

# --- helpers ---
_has_rg = shutil.which("rg") is not None
skip_if_no_rg = pytest.mark.skipif(
    not _has_rg, reason="ripgrep (rg) not installed — SearchFilesTool tests skipped"
)


@pytest.fixture
def temp_dir() -> Path:
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


# ---------------------------------------------------------------------------
# ReadFileTool
# ---------------------------------------------------------------------------
class TestReadFileTool:
    async def test_reads_text_file(self, temp_dir: Path) -> None:
        f = temp_dir / "hello.txt"
        f.write_text("Hello, world!")
        tool = ReadFileTool()
        result = await tool.run(str(f))
        assert result == "Hello, world!"

    async def test_file_not_found(self) -> None:
        tool = ReadFileTool()
        with pytest.raises(ToolError, match="not found"):
            await tool.run("/nonexistent/file.txt")


# ---------------------------------------------------------------------------
# WriteFileTool
# ---------------------------------------------------------------------------
class TestWriteFileTool:
    async def test_writes_new_file(self, temp_dir: Path) -> None:
        target = temp_dir / "new.txt"
        tool = WriteFileTool()
        result = await tool.run(str(target), "content")
        assert target.read_text() == "content"
        assert "Wrote 7 bytes" in result

    async def test_creates_parent_dirs(self, temp_dir: Path) -> None:
        target = temp_dir / "a" / "b" / "c.txt"
        tool = WriteFileTool()
        await tool.run(str(target), "deep")
        assert target.exists()
        assert target.read_text() == "deep"


# ---------------------------------------------------------------------------
# ListFilesTool
# ---------------------------------------------------------------------------
class TestListFilesTool:
    async def test_lists_directory(self, temp_dir: Path) -> None:
        (temp_dir / "file_a.py").write_text("a")
        (temp_dir / "file_b.py").write_text("b")
        tool = ListFilesTool()
        result = await tool.run(str(temp_dir))
        assert "file_a.py" in result
        assert "file_b.py" in result
        assert "2 entries" in result

    async def test_empty_directory(self, temp_dir: Path) -> None:
        tool = ListFilesTool()
        result = await tool.run(str(temp_dir))
        assert "empty" in result


# ---------------------------------------------------------------------------
# SearchFilesTool
# ---------------------------------------------------------------------------
class TestSearchFilesTool:
    @skip_if_no_rg
    async def test_finds_pattern(self, temp_dir: Path) -> None:
        (temp_dir / "notes.txt").write_text("apple banana cherry")
        tool = SearchFilesTool()
        result = await tool.run("banana", path=str(temp_dir))
        assert "1 match" in result
        assert "notes.txt" in result

    @skip_if_no_rg
    async def test_no_match(self, temp_dir: Path) -> None:
        (temp_dir / "notes.txt").write_text("apple")
        tool = SearchFilesTool()
        result = await tool.run("zzzzz", path=str(temp_dir))
        assert "No matches" in result


# ---------------------------------------------------------------------------
# DeleteFileTool
# ---------------------------------------------------------------------------
class TestDeleteFileTool:
    async def test_deletes_file(self, temp_dir: Path) -> None:
        f = temp_dir / "trash.txt"
        f.write_text("bye")
        tool = DeleteFileTool()
        result = await tool.run(str(f))
        assert not f.exists()
        assert "Deleted" in result

    async def test_not_found(self) -> None:
        tool = DeleteFileTool()
        with pytest.raises(ToolError, match="Not found"):
            await tool.run("/nonexistent/file.txt")
