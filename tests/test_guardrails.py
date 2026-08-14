"""Tests for GuardrailMiddleware — the configurable tool-call policy layer."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.tools.base import ToolError
from src.tools.middleware import (
    DEFAULT_MAX_RESULT_CHARS,
    GuardrailMiddleware,
    ToolPipeline,
)


class _CountingDispatch:
    """Async dispatch fn that records how many times it was invoked."""

    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, name: str, args: dict) -> str:
        self.calls += 1
        return "ok"


# --- path confinement ------------------------------------------------------
class TestPathConfinement:
    async def test_path_inside_work_dir_passes(self, tmp_path: Path) -> None:
        mw = GuardrailMiddleware(work_dir=tmp_path)
        args = {"path": str(tmp_path / "f.txt")}

        result = await mw.before("read_file", args)

        assert result == args  # unchanged

    async def test_etc_passwd_blocked(self, tmp_path: Path) -> None:
        mw = GuardrailMiddleware(work_dir=tmp_path)

        with pytest.raises(ToolError, match="Path traversal"):
            await mw.before("read_file", {"path": "/etc/passwd"})

    async def test_write_file_outside_base_blocked(self, tmp_path: Path) -> None:
        mw = GuardrailMiddleware(work_dir=tmp_path)

        with pytest.raises(ToolError, match="Path traversal"):
            await mw.before("write_file", {"path": "/tmp/evil.txt", "content": "x"})

    async def test_move_file_dest_outside_blocked(self, tmp_path: Path) -> None:
        mw = GuardrailMiddleware(work_dir=tmp_path)

        with pytest.raises(ToolError, match="Path traversal"):
            await mw.before("move_file", {"source": str(tmp_path / "a.txt"), "dest": "/tmp/b.txt"})

    async def test_relative_escape_blocked(self, tmp_path: Path) -> None:
        mw = GuardrailMiddleware(work_dir=tmp_path)

        with pytest.raises(ToolError, match="Path traversal"):
            await mw.before("read_file", {"path": str(tmp_path / ".." / ".." / "etc" / "passwd")})

    async def test_absent_optional_path_passes(self, tmp_path: Path) -> None:
        mw = GuardrailMiddleware(work_dir=tmp_path)

        result = await mw.before("list_files", {"path": None})
        assert result == {"path": None}

    async def test_search_files_without_path_passes(self, tmp_path: Path) -> None:
        mw = GuardrailMiddleware(work_dir=tmp_path)

        result = await mw.before("search_files", {"pattern": "TODO"})
        assert result == {"pattern": "TODO"}

    async def test_run_shell_work_dir_outside_blocked(self, tmp_path: Path) -> None:
        mw = GuardrailMiddleware(work_dir=tmp_path)

        with pytest.raises(ToolError, match="Path traversal"):
            await mw.before("run_shell", {"command": "ls", "work_dir": "/etc"})

    async def test_run_shell_work_dir_inside_passes(self, tmp_path: Path) -> None:
        mw = GuardrailMiddleware(work_dir=tmp_path)

        result = await mw.before("run_shell", {"command": "ls", "work_dir": str(tmp_path)})
        assert result == {"command": "ls", "work_dir": str(tmp_path)}

    @pytest.mark.parametrize("tool_name", ["json_read", "json_write", "csv_read", "csv_aggregate"])
    async def test_day_to_day_path_tools_inside_passes(
        self, tmp_path: Path, tool_name: str
    ) -> None:
        mw = GuardrailMiddleware(work_dir=tmp_path)
        args = {"path": str(tmp_path / "f.json")}

        result = await mw.before(tool_name, args)

        assert result == args

    @pytest.mark.parametrize("tool_name", ["json_read", "json_write", "csv_read", "csv_aggregate"])
    async def test_day_to_day_path_tools_escape_blocked(
        self, tmp_path: Path, tool_name: str
    ) -> None:
        mw = GuardrailMiddleware(work_dir=tmp_path)

        with pytest.raises(ToolError, match="Path traversal"):
            await mw.before(tool_name, {"path": "/etc/passwd"})

    @pytest.mark.parametrize("tool_name", ["copy_file", "file_info", "mkdir"])
    async def test_new_file_tools_inside_passes(self, tmp_path: Path, tool_name: str) -> None:
        mw = GuardrailMiddleware(work_dir=tmp_path)
        args = (
            {"source": str(tmp_path / "a.txt"), "dest": str(tmp_path / "b.txt")}
            if tool_name == "copy_file"
            else {"path": str(tmp_path / "f.txt")}
        )

        result = await mw.before(tool_name, args)

        assert result == args

    @pytest.mark.parametrize("tool_name", ["copy_file", "file_info", "mkdir"])
    async def test_new_file_tools_escape_blocked(self, tmp_path: Path, tool_name: str) -> None:
        mw = GuardrailMiddleware(work_dir=tmp_path)
        args = (
            {"source": str(tmp_path / "a.txt"), "dest": "/tmp/evil.txt"}
            if tool_name == "copy_file"
            else {"path": "/etc/passwd"}
        )

        with pytest.raises(ToolError, match="Path traversal"):
            await mw.before(tool_name, args)


# --- shell deny-prefixes ----------------------------------------------------
class TestShellDenyPrefixes:
    async def test_rm_rf_root_blocked(self, tmp_path: Path) -> None:
        mw = GuardrailMiddleware(work_dir=tmp_path)

        with pytest.raises(ToolError, match="Guardrail blocked command"):
            await mw.before("run_shell", {"command": "rm -rf /"})

    async def test_benign_command_passes(self, tmp_path: Path) -> None:
        mw = GuardrailMiddleware(work_dir=tmp_path)

        result = await mw.before("run_shell", {"command": "ls -la"})
        assert result == {"command": "ls -la"}

    async def test_custom_deny_prefixes_override(self, tmp_path: Path) -> None:
        mw = GuardrailMiddleware(work_dir=tmp_path, deny_prefixes=("git push",))

        with pytest.raises(ToolError, match="git push"):
            await mw.before("run_shell", {"command": "git push origin main"})

        # the default list no longer applies
        result = await mw.before("run_shell", {"command": "rm -rf /"})
        assert result == {"command": "rm -rf /"}


# --- output cap -------------------------------------------------------------
class TestOutputCap:
    async def test_long_result_truncated(self) -> None:
        mw = GuardrailMiddleware(work_dir=Path("."), max_result_chars=10)

        result = await mw.after("read_file", "a" * 100, None)

        assert len(result) > 10  # marker appended
        assert result.startswith("a" * 10)
        assert "[truncated: 100 chars, max 10]" in result

    async def test_short_result_unchanged(self) -> None:
        mw = GuardrailMiddleware(work_dir=Path("."), max_result_chars=10)

        result = await mw.after("read_file", "abc", None)

        assert result == "abc"

    async def test_zero_cap_disables_truncation(self) -> None:
        mw = GuardrailMiddleware(work_dir=Path("."), max_result_chars=0)

        result = await mw.after("read_file", "a" * 500, None)

        assert result == "a" * 500

    async def test_error_result_not_truncated(self) -> None:
        mw = GuardrailMiddleware(work_dir=Path("."), max_result_chars=10)
        exc = RuntimeError("boom")

        result = await mw.after("read_file", "a" * 100, exc)

        assert result == "a" * 100


# --- toggle -----------------------------------------------------------------
class TestToggle:
    async def test_disabled_skips_path_check(self, tmp_path: Path) -> None:
        mw = GuardrailMiddleware(work_dir=tmp_path, enabled=False)

        result = await mw.before("read_file", {"path": "/etc/passwd"})

        assert result == {"path": "/etc/passwd"}

    async def test_disabled_skips_deny_check(self, tmp_path: Path) -> None:
        mw = GuardrailMiddleware(work_dir=tmp_path, enabled=False)

        result = await mw.before("run_shell", {"command": "rm -rf /"})

        assert result == {"command": "rm -rf /"}

    async def test_disabled_skips_truncation(self) -> None:
        mw = GuardrailMiddleware(work_dir=Path("."), enabled=False, max_result_chars=10)

        result = await mw.after("read_file", "a" * 100, None)

        assert result == "a" * 100


# --- pipeline integration ---------------------------------------------------
class TestPipelineIntegration:
    async def test_blocked_call_never_dispatches(self, tmp_path: Path) -> None:
        dispatch = _CountingDispatch()
        pipeline = ToolPipeline(middleware=[GuardrailMiddleware(work_dir=tmp_path)])

        with pytest.raises(ToolError, match="Path traversal"):
            await pipeline.run(dispatch, "read_file", {"path": "/etc/passwd"})

        assert dispatch.calls == 0

    async def test_allowed_call_dispatches_and_returns(self, tmp_path: Path) -> None:
        dispatch = _CountingDispatch()
        pipeline = ToolPipeline(middleware=[GuardrailMiddleware(work_dir=tmp_path)])

        result = await pipeline.run(dispatch, "read_file", {"path": str(tmp_path / "f.txt")})

        assert result == "ok"
        assert dispatch.calls == 1


# --- defaults ---------------------------------------------------------------
class TestDefaults:
    def test_default_result_cap(self) -> None:
        assert DEFAULT_MAX_RESULT_CHARS == 20_000
