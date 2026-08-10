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

from src.memory.sqlite_store import DEFAULT_DB_PATH


def _default_db_path() -> Path:
    """Return the shared database path — ``TASKFLOW_DB`` env override or the default."""
    configured = os.getenv("TASKFLOW_DB")
    if configured:
        return Path(configured).expanduser()
    return DEFAULT_DB_PATH


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

    # --- Prompt caching (Anthropic ephemeral cache_control breakpoints) ---
    prompt_caching_enabled: bool = field(
        default_factory=lambda: os.getenv("PROMPT_CACHING_ENABLED", "1").lower()
        in ("1", "true", "yes")
    )

    # --- Guardrails (policy layer gating tool calls) ---
    guardrails_enabled: bool = field(
        default_factory=lambda: os.getenv("GUARDRAILS_ENABLED", "1").lower()
        in ("1", "true", "yes")
    )
    max_tool_result_chars: int = field(
        default_factory=lambda: int(os.getenv("MAX_TOOL_RESULT_CHARS", "20000"))
    )

    # --- Paths ---
    work_dir: Path = field(default_factory=lambda: Path(os.getenv("AGENT_WORK_DIR", ".")).resolve())
    memory_dir: Path = field(default_factory=lambda: Path.home() / ".taskflow" / "memory")
    db_path: Path = field(default_factory=_default_db_path)

    # --- Limits ---
    max_tool_calls_per_turn: int = field(
        default_factory=lambda: int(os.getenv("MAX_TOOL_CALLS", "25"))
    )
    max_parallel_tool_calls: int = field(
        default_factory=lambda: int(os.getenv("MAX_PARALLEL_TOOL_CALLS", "5"))
    )
    max_history_tokens: int = field(
        default_factory=lambda: int(os.getenv("MAX_HISTORY_TOKENS", "100000"))
    )
    summary_threshold_tokens: int = field(
        default_factory=lambda: int(os.getenv("MEMORY_SUMMARY_THRESHOLD", "60000"))
    )
    memory_inject_facts: int = field(
        default_factory=lambda: int(os.getenv("MEMORY_INJECT_FACTS", "5"))
    )

    # --- Usage & cost ---
    max_cost_usd: float = field(
        default_factory=lambda: float(os.getenv("MAX_COST_USD", "0"))
    )

    # --- Logging ---
    log_level: str = field(default_factory=lambda: os.getenv("AGENT_LOG_LEVEL", "INFO").upper())
    log_file: str = field(default_factory=lambda: os.getenv("AGENT_LOG_FILE", ""))

    def __post_init__(self) -> None:
        """Validate and prepare the configuration."""
        _validate_settings(self)

        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def is_ready(self) -> bool:
        return bool(self.anthropic_api_key)


def _validate_settings(s: Settings) -> None:
    """Validate settings values, raising ``ValueError`` for invalid ones."""
    if not 0 <= s.safety_level <= 3:
        raise ValueError(f"SAFETY_LEVEL must be 0-3, got {s.safety_level}")

    if s.max_tool_calls_per_turn < 1:
        raise ValueError(f"MAX_TOOL_CALLS must be >= 1, got {s.max_tool_calls_per_turn}")

    if s.max_parallel_tool_calls < 1:
        raise ValueError(
            f"MAX_PARALLEL_TOOL_CALLS must be >= 1 (1 disables parallelism), "
            f"got {s.max_parallel_tool_calls}"
        )

    if s.max_history_tokens < 1000:
        raise ValueError(f"MAX_HISTORY_TOKENS must be >= 1000, got {s.max_history_tokens}")

    if s.summary_threshold_tokens < 0:
        raise ValueError(
            f"MEMORY_SUMMARY_THRESHOLD must be >= 0 (0 disables), got {s.summary_threshold_tokens}"
        )

    if s.memory_inject_facts < 0:
        raise ValueError(
            f"MEMORY_INJECT_FACTS must be >= 0 (0 disables injection), "
            f"got {s.memory_inject_facts}"
        )

    if s.max_tool_result_chars < 0:
        raise ValueError(
            f"MAX_TOOL_RESULT_CHARS must be >= 0 (0 disables truncation), got {s.max_tool_result_chars}"
        )

    if s.max_cost_usd < 0:
        raise ValueError(
            f"MAX_COST_USD must be >= 0 (0 disables the cap), got {s.max_cost_usd}"
        )

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
