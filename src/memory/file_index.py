"""Lightweight file-system index for quick lookups without full traversal."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class FileIndex:
    """Caches file layout so the agent can answer "what's where" quickly.

    The index is stored as JSON on disk and refreshed on demand.
    """

    def __init__(self, index_path: Path | None = None) -> None:
        self._index_path = index_path or Path.home() / ".taskflow" / "file_index.json"
        self._data: dict[str, dict[str, Any]] = {}
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        if self._index_path.exists():
            try:
                self._data = json.loads(self._index_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._data = {}

    def _save(self) -> None:
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        self._index_path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    def refresh(self, root: Path | str) -> str:
        """Walk *root* and build an index of all files.

        Returns a summary string for the LLM.
        """
        root = Path(root).expanduser().resolve()
        if not root.is_dir():
            return f"Not a directory: {root}"

        entries: dict[str, dict[str, Any]] = {}
        total_size = 0
        file_count = 0
        dir_count = 0

        for p in root.rglob("*"):
            try:
                rel = str(p.relative_to(root))
                if p.is_dir():
                    entries[rel] = {"type": "dir"}
                    dir_count += 1
                elif p.is_file():
                    s = p.stat()
                    entries[rel] = {
                        "type": "file",
                        "size": s.st_size,
                        "modified": datetime.fromtimestamp(s.st_mtime).isoformat(),
                    }
                    total_size += s.st_size
                    file_count += 1
            except (OSError, ValueError):
                continue

        self._data[str(root)] = {
            "entries": entries,
            "file_count": file_count,
            "dir_count": dir_count,
            "total_size": total_size,
            "indexed_at": datetime.now().isoformat(),
        }
        self._save()

        return f"Indexed {root}: {file_count} files, {dir_count} directories, {total_size:,} bytes"

    # ------------------------------------------------------------------
    def query(self, path: str | None = None) -> str:
        """Return a human-readable summary of what's in the index."""
        if not self._data:
            return "Index is empty — run refresh() first."

        if path:
            key = str(Path(path).expanduser().resolve())
            info = self._data.get(key)
            if info is None:
                return f"No index entry for {key}"
            return (
                f"Indexed at {info['indexed_at']}: "
                f"{info['file_count']} files, {info['dir_count']} dirs, "
                f"{info['total_size']:,} bytes"
            )

        lines = ["Indexed roots:"]
        for root, info in self._data.items():
            lines.append(
                f"  {root}: {info['file_count']} files, "
                f"{info['dir_count']} dirs, {info['total_size']:,} bytes"
            )
        return "\n".join(lines)
