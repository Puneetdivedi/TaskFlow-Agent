"""Tests for the file-system index."""

from __future__ import annotations

import json
from pathlib import Path

from src.memory.file_index import FileIndex


class TestFileIndex:
    def test_query_empty_index(self, tmp_path: Path) -> None:
        index_path = tmp_path / "index.json"
        idx = FileIndex(index_path=index_path)
        result = idx.query()
        assert "empty" in result

    def test_refresh_empty_directory(self, tmp_path: Path) -> None:
        idx = FileIndex(index_path=tmp_path / "index.json")
        result = idx.refresh(tmp_path)
        assert "Indexed" in result
        assert "0 files" in result

    def test_refresh_tracks_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("hello")
        (tmp_path / "b.txt").write_text("world")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "c.txt").write_text("deep")

        idx = FileIndex(index_path=tmp_path / "index.json")
        result = idx.refresh(tmp_path)
        assert "3 files" in result
        assert "1 directories" in result

    def test_refresh_creates_save_file(self, tmp_path: Path) -> None:
        index_file = tmp_path / "index.json"
        idx = FileIndex(index_path=index_file)
        idx.refresh(tmp_path)
        assert index_file.exists()
        data = json.loads(index_file.read_text())
        assert str(tmp_path) in data

    def test_query_after_refresh(self, tmp_path: Path) -> None:
        (tmp_path / "data.txt").write_text("data")
        idx = FileIndex(index_path=tmp_path / "index.json")
        idx.refresh(tmp_path)
        result = idx.query(str(tmp_path))
        assert "1 files" in result or "1 file" in result

    def test_corrupt_index_does_not_crash(self, tmp_path: Path) -> None:
        index_file = tmp_path / "index.json"
        index_file.write_text("this is not json")
        idx = FileIndex(index_path=index_file)
        # Should have gracefully handled the corrupt file
        result = idx.query()
        assert "empty" in result

    def test_multiple_roots(self, tmp_path: Path) -> None:
        idx = FileIndex(index_path=tmp_path / "index.json")
        d1 = tmp_path / "dir1"
        d2 = tmp_path / "dir2"
        d1.mkdir()
        d2.mkdir()
        (d1 / "f1.txt").write_text("a")
        (d2 / "f2.txt").write_text("b")

        idx.refresh(d1)
        idx.refresh(d2)

        result = idx.query()
        assert "Indexed roots" in result
        assert "dir1" in result
        assert "dir2" in result
