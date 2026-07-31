"""Persistent session storage — saves conversation threads as JSON files.

Sessions live as ``<name>.json`` files inside a session directory (default
``~/.taskflow/sessions``), mirroring the ``FileIndex`` JSON persistence
pattern.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from src.interfaces.session_store import SessionInfo

logger = logging.getLogger(__name__)

# Names are slugs: start with an alphanumeric, then letters/digits/._-.
# This blocks path traversal ("../x", "a/b"), hidden files (".x", ".."),
# and anything a filesystem could reject.
_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_MAX_NAME_LEN = 64


class SessionStore:
    """JSON-backed store for named conversation sessions."""

    def __init__(self, session_dir: Path | None = None) -> None:
        self._dir = session_dir or Path.home() / ".taskflow" / "sessions"
        self._dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def _validate_name(self, name: str) -> None:
        if not name:
            raise ValueError("Session name must not be empty")
        if len(name) > _MAX_NAME_LEN:
            raise ValueError(f"Session name too long (max {_MAX_NAME_LEN} chars): {name!r}")
        if not _NAME_RE.fullmatch(name):
            raise ValueError(
                "Session name must contain only letters, digits, '.', '_', '-' "
                f"and start with a letter or digit — got {name!r}"
            )

    def _path_for(self, name: str) -> Path:
        self._validate_name(name)
        path = self._dir / f"{name}.json"
        if path.parent.resolve() != self._dir.resolve():
            raise ValueError(f"Session name escapes the session directory: {name!r}")
        return path

    # ------------------------------------------------------------------
    def save(self, name: str, messages: list[dict[str, Any]]) -> None:
        """Persist *messages* under the session *name*."""
        self._validate_name(name)
        payload = {
            "name": name,
            "saved_at": datetime.now().isoformat(),
            "message_count": len(messages),
            "messages": messages,
        }
        path = self._dir / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Saved session %r with %d message(s)", name, len(messages))

    def load(self, name: str) -> list[dict[str, Any]]:
        """Return the messages saved under *name*.

        Raises ``KeyError`` if the session does not exist and ``ValueError``
        if the stored file is corrupted.
        """
        path = self._path_for(name)
        if not path.exists():
            raise KeyError(f"Session {name!r} not found")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(f"Session {name!r} is corrupted: {exc}") from exc
        messages = payload.get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"Session {name!r} is corrupted: missing messages list")
        return messages

    def delete(self, name: str) -> None:
        """Remove the saved session *name*.

        Raises ``KeyError`` if the session does not exist.
        """
        path = self._path_for(name)
        if not path.exists():
            raise KeyError(f"Session {name!r} not found")
        path.unlink()
        logger.info("Deleted session %r", name)

    def exists(self, name: str) -> bool:
        """Return whether a session named *name* exists."""
        try:
            return self._path_for(name).exists()
        except ValueError:
            return False

    def list(self) -> list[SessionInfo]:
        """Return metadata for all saved sessions, newest first.

        Corrupted or malformed files are skipped so one bad file never
        breaks the whole listing.
        """
        infos: list[SessionInfo] = []
        for path in self._dir.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                name = payload.get("name")
                saved_at = payload.get("saved_at")
                message_count = payload.get("message_count")
                if not isinstance(name, str) or not isinstance(saved_at, str):
                    continue
                infos.append(
                    SessionInfo(
                        name=name,
                        message_count=int(message_count) if isinstance(message_count, int) else 0,
                        updated_at=saved_at,
                        file_path=path,
                    )
                )
            except (json.JSONDecodeError, OSError, ValueError):
                logger.warning("Skipping corrupted session file: %s", path)
                continue
        infos.sort(key=lambda info: info.updated_at, reverse=True)
        return infos

    def latest(self) -> str | None:
        """Return the name of the most recently saved session, or ``None``."""
        infos = self.list()
        return infos[0].name if infos else None
