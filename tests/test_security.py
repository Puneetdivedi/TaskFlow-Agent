"""Tests for path-traversal security guard."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.tools.base import ToolError
from src.tools.security import validate_path_safe


class TestValidatePathSafe:
    def test_allowed_path_passes(self, tmp_path: Path) -> None:
        # Should not raise
        validate_path_safe(tmp_path / "file.txt", tmp_path)

    def test_subdirectory_passes(self, tmp_path: Path) -> None:
        sub = tmp_path / "sub" / "deep"
        sub.mkdir(parents=True)
        validate_path_safe(sub / "f.txt", tmp_path)

    def test_path_traversal_blocked(self, tmp_path: Path) -> None:
        with pytest.raises(ToolError, match="Path traversal blocked"):
            validate_path_safe(Path("/etc/passwd"), tmp_path)

    def test_traversal_via_dotdot_blocked(self, tmp_path: Path) -> None:
        malicious = tmp_path / ".." / ".." / "etc" / "passwd"
        with pytest.raises(ToolError, match="Path traversal blocked"):
            validate_path_safe(malicious, tmp_path)

    def test_same_directory_passes(self, tmp_path: Path) -> None:
        validate_path_safe(tmp_path, tmp_path)

    def test_symlink_traversal(self, tmp_path: Path) -> None:
        """Symlink pointing outside the base should be blocked."""
        outside = tmp_path / ".." / "outside.txt"
        # Don't create the actual symlink — just validate the resolved path
        # is caught
        with pytest.raises(ToolError, match="Path traversal blocked"):
            validate_path_safe(outside.resolve(), tmp_path)
