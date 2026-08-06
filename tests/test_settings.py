"""Tests for configuration settings and validation."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from config.settings import Settings
from src.memory.sqlite_store import DEFAULT_DB_PATH


class TestSettings:
    def test_default_values(self) -> None:
        s = Settings()
        assert s.max_tool_calls_per_turn == 25
        assert s.max_history_tokens == 100_000
        assert s.log_level == "INFO"
        assert s.safety_level == 1

    def test_is_ready_false_without_api_key(self) -> None:
        # Ensure env doesn't have a key
        old = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            s = Settings()
            assert s.is_ready is False
        finally:
            if old is not None:
                os.environ["ANTHROPIC_API_KEY"] = old

    def test_safety_level_validation(self) -> None:
        for valid in (0, 1, 2, 3):
            s = Settings(safety_level=valid)
            assert s.safety_level == valid

    def test_safety_level_too_low(self) -> None:
        with pytest.raises(ValueError, match="SAFETY_LEVEL must be 0-3"):
            Settings(safety_level=-1)

    def test_safety_level_too_high(self) -> None:
        with pytest.raises(ValueError, match="SAFETY_LEVEL must be 0-3"):
            Settings(safety_level=5)

    def test_max_tool_calls_positive(self) -> None:
        with pytest.raises(ValueError, match="MAX_TOOL_CALLS must be >= 1"):
            Settings(max_tool_calls_per_turn=0)

    def test_max_history_tokens_minimum(self) -> None:
        with pytest.raises(ValueError, match="MAX_HISTORY_TOKENS must be >= 1000"):
            Settings(max_history_tokens=500)

    def test_log_level_validation(self) -> None:
        with pytest.raises(ValueError, match="AGENT_LOG_LEVEL must be one of"):
            Settings(log_level="TRACE")

    def test_valid_log_levels(self) -> None:
        for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            s = Settings(log_level=level)
            assert s.log_level == level

    def test_work_dir_resolves(self) -> None:
        s = Settings()
        assert isinstance(s.work_dir, Path)
        assert s.work_dir.is_absolute()

    def test_memory_dir_created(self, tmp_path: Path) -> None:
        """memory_dir should be created in __post_init__."""
        mem_dir = tmp_path / ".taskflow" / "memory"
        Settings(memory_dir=mem_dir)
        assert mem_dir.exists()
        assert mem_dir.is_dir()

    def test_db_path_defaults_to_taskflow_db(self) -> None:
        s = Settings()
        assert s.db_path == DEFAULT_DB_PATH

    def test_db_path_honors_env_override(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        custom = tmp_path / "custom" / "taskflow.db"
        monkeypatch.setenv("TASKFLOW_DB", str(custom))
        assert Settings().db_path == custom

    def test_db_path_parent_created(self, tmp_path: Path) -> None:
        """The db directory should be created in __post_init__."""
        db_path = tmp_path / ".taskflow" / "taskflow.db"
        Settings(db_path=db_path)
        assert db_path.parent.exists()
        assert db_path.parent.is_dir()
