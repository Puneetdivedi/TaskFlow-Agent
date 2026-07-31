"""Protocol for filesystem index backends."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class IFileIndex(Protocol):
    """Interface for caching filesystem layout information."""

    def refresh(self, root: Path | str) -> str:
        """Walk *root* and build/update the index.

        Returns a human-readable summary string.
        """
        ...

    def query(self, path: str | None = None) -> str:
        """Return a summary of what's in the index for *path* (or everything)."""
        ...
