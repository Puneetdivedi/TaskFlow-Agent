"""Application configuration — loads from environment / .env file.

Usage::

    from config.settings import init_settings
    settings = init_settings()
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


def _default_session_dir() -> Path:
    """Return the session directory — ``SESSION_DIR`` env override or the default."""
    configured = os.getenv("SESSION_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".taskflow" / "sessions"


@dataclass
class Settings:
    """Immutable-ish settings container.

    All fields are populated from environment variables (optionally
    loaded from a ``.env`` file by :func:`init_settings`).
    """

    # --- Anthropic ---
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    anthropic_model: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-20250611")
    )

    # --- Safety ---
    safety_level: int = field(default_factory=lambda: int(os.getenv("SAFETY_LEVEL", "1")))

    # --- Paths ---
    work_dir: Path = field(default_factory=lambda: Path(os.getenv("AGENT_WORK_DIR", ".")).resolve())
    memory_dir: Path = field(default_factory=lambda: Path.home() / ".taskflow" / "memory")
    session_dir: Path = field(default_factory=_default_session_dir)

    # --- Limits ---
    max_tool_calls_per_turn: int = field(
        default_factory=lambda: int(os.getenv("MAX_TOOL_CALLS", "25"))
    )
    max_history_tokens: int = field(
        default_factory=lambda: int(os.getenv("MAX_HISTORY_TOKENS", "100000"))
    )

    # --- Logging ---
    log_level: str = field(default_factory=lambda: os.getenv("AGENT_LOG_LEVEL", "INFO").upper())
    log_file: str = field(default_factory=lambda: os.getenv("AGENT_LOG_FILE", ""))

    def __post_init__(self) -> None:
        """Validate and prepare the configuration."""
        _validate_settings(self)

        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.session_dir.mkdir(parents=True, exist_ok=True)

    @property
    def is_ready(self) -> bool:
        return bool(self.anthropic_api_key)


def _validate_settings(s: Settings) -> None:
    """Validate settings values, raising ``ValueError`` for invalid ones."""
    if not 0 <= s.safety_level <= 3:
        raise ValueError(f"SAFETY_LEVEL must be 0-3, got {s.safety_level}")

    if s.max_tool_calls_per_turn < 1:
        raise ValueError(f"MAX_TOOL_CALLS must be >= 1, got {s.max_tool_calls_per_turn}")

    if s.max_history_tokens < 1000:
        raise ValueError(f"MAX_HISTORY_TOKENS must be >= 1000, got {s.max_history_tokens}")

    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if s.log_level not in valid_levels:
        raise ValueError(
            f"AGENT_LOG_LEVEL must be one of {', '.join(sorted(valid_levels))}, got {s.log_level!r}"
        )


_SETTINGS_CACHE: Settings | None = None


def init_settings() -> Settings:
    """Initialise settings from environment / ``.env`` file.

    Calling this function multiple times returns the *same* cached
    ``Settings`` instance (singleton).
    """
    global _SETTINGS_CACHE  # noqa: PLW0603
    if _SETTINGS_CACHE is not None:
        return _SETTINGS_CACHE

    load_dotenv()
    _SETTINGS_CACHE = Settings()
    return _SETTINGS_CACHE
