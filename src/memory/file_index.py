"""Lightweight file-system index for quick lookups without full traversal."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.memory.sqlite_store import INDEX_SCHEMA, SQLiteStore


class FileIndex(SQLiteStore):
    """Caches file layout so the agent can answer "what's where" quickly.

    The index is stored in the ``index_roots`` table of the shared TaskFlow
    database and refreshed on demand.
    """

    _SCHEMA = INDEX_SCHEMA

    def __init__(self, db_path: Path | None = None) -> None:
        super().__init__(db_path)

    # ------------------------------------------------------------------
    def refresh(self, root: Path | str) -> str:
        """Walk *root* and build an index of all files.

        Returns a summary string for the LLM.
        """
        root = Path(root).expanduser().resolve()
        if not root.is_dir():
            return f"Not a directory: {root}"

        entries: dict[str, dict[str, object]] = {}
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

        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO index_roots "
                "(root, file_count, dir_count, total_size, indexed_at, entries) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    str(root),
                    file_count,
                    dir_count,
                    total_size,
                    datetime.now().isoformat(),
                    json.dumps(entries),
                ),
            )

        return f"Indexed {root}: {file_count} files, {dir_count} directories, {total_size:,} bytes"

    # ------------------------------------------------------------------
    def query(self, path: str | None = None) -> str:
        """Return a human-readable summary of what's in the index."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT root, file_count, dir_count, total_size, indexed_at FROM index_roots"
            ).fetchall()
        if not rows:
            return "Index is empty — run refresh() first."

        if path:
            key = str(Path(path).expanduser().resolve())
            for row in rows:
                if row["root"] == key:
                    return (
                        f"Indexed at {row['indexed_at']}: "
                        f"{row['file_count']} files, {row['dir_count']} dirs, "
                        f"{row['total_size']:,} bytes"
                    )
            return f"No index entry for {key}"

        lines = ["Indexed roots:"]
        for row in rows:
            lines.append(
                f"  {row['root']}: {row['file_count']} files, "
                f"{row['dir_count']} dirs, {row['total_size']:,} bytes"
            )
        return "\n".join(lines)
