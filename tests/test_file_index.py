"""Tests for the file-system index."""

from __future__ import annotations

from pathlib import Path

from src.memory.file_index import FileIndex


class TestFileIndex:
    def test_query_empty_index(self, tmp_path: Path) -> None:
        idx = FileIndex(db_path=tmp_path / "index.db")
        result = idx.query()
        assert "empty" in result

    def test_refresh_empty_directory(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        root.mkdir()
        idx = FileIndex(db_path=tmp_path / "index.db")
        result = idx.refresh(root)
        assert "Indexed" in result
        assert "0 files" in result

    def test_refresh_tracks_files(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        root.mkdir()
        (root / "a.txt").write_text("hello")
        (root / "b.txt").write_text("world")
        (root / "sub").mkdir()
        (root / "sub" / "c.txt").write_text("deep")

        idx = FileIndex(db_path=tmp_path / "index.db")
        result = idx.refresh(root)
        assert "3 files" in result
        assert "1 directories" in result

    def test_refresh_creates_save_file(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        root.mkdir()
        db_file = tmp_path / "index.db"
        idx = FileIndex(db_path=db_file)
        idx.refresh(root)
        assert db_file.exists()
        result = idx.query(str(root))
        assert "files" in result

    def test_query_after_refresh(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        root.mkdir()
        (root / "data.txt").write_text("data")
        idx = FileIndex(db_path=tmp_path / "index.db")
        idx.refresh(root)
        result = idx.query(str(root))
        assert "1 files" in result or "1 file" in result

    def test_corrupt_index_does_not_crash(self, tmp_path: Path) -> None:
        db_file = tmp_path / "index.db"
        db_file.write_text("this is not a sqlite database")
        idx = FileIndex(db_path=db_file)
        # Should have gracefully rebuilt the corrupt database.
        result = idx.query()
        assert "empty" in result

    def test_multiple_roots(self, tmp_path: Path) -> None:
        idx = FileIndex(db_path=tmp_path / "index.db")
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
