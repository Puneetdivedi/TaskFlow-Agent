"""Shared security utilities for tool implementations.

Currently provides:
- :func:`validate_path_safe` — guards against path-traversal attacks.
"""

from __future__ import annotations

from pathlib import Path

from src.tools.base import ToolError


def validate_path_safe(path: Path, allowed_base: Path) -> None:
    """Verify that *path* resolves within *allowed_base*.

    Raises :exc:`ToolError` with a descriptive message if the resolved
    path falls outside the allowed base directory — preventing
    ``../../etc/passwd`` style traversal.
    """
    resolved = path.expanduser().resolve()
    base = allowed_base.expanduser().resolve()

    try:
        resolved.relative_to(base)
    except ValueError:
        raise ToolError(
            f"Path traversal blocked: {resolved} is outside the "
            f"allowed base directory ({base})"
        ) from None
