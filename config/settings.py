"""Application configuration — loads from .env / environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    # --- Anthropic ---
    anthropic_api_key: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", "")
    )
    anthropic_model: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-20250611")
    )

    # --- Safety ---
    safety_level: int = field(
        default_factory=lambda: int(os.getenv("SAFETY_LEVEL", "1"))
    )

    # --- Paths ---
    work_dir: Path = field(
        default_factory=lambda: Path(os.getenv("AGENT_WORK_DIR", ".")).resolve()
    )
    memory_dir: Path = field(
        default_factory=lambda: Path.home() / ".taskflow" / "memory"
    )

    # --- Limits ---
    max_tool_calls_per_turn: int = 25
    max_history_tokens: int = 100_000

    def __post_init__(self) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    @property
    def is_ready(self) -> bool:
        return bool(self.anthropic_api_key)


SETTINGS = Settings()
