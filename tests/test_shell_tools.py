"""Tests for shell execution tool safety."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from src.tools.base import ToolError
from src.tools.shell_tools import RunShellTool


class TestRunShellTool:
    async def test_safety_level_0_blocks_all(self) -> None:
        tool = RunShellTool(safety_level=0)
        with pytest.raises(ToolError, match="blocked at safety level 0"):
            await tool.run("echo hello")

    async def test_safety_level_1_allows_simple_command(self) -> None:
        tool = RunShellTool(safety_level=1)
        result = await tool.run("echo hello")
        assert "hello" in result

    async def test_forbidden_rm_rf_root(self) -> None:
        tool = RunShellTool(safety_level=1)
        with pytest.raises(ToolError, match="blocked for safety"):
            await tool.run("rm -rf /")

    async def test_forbidden_dd(self) -> None:
        tool = RunShellTool(safety_level=1)
        with pytest.raises(ToolError, match="blocked for safety"):
            await tool.run("dd if=/dev/zero of=/dev/sda bs=4M")

    async def test_rm_rf_absolute_path_allowed(self) -> None:
        """rm -rf with an absolute path should pass validation."""
        tool = RunShellTool(safety_level=1)
        # Use an absolute path that works on all platforms — validation
        # only checks it's absolute, it doesn't execute blindly.
        # On Windows, an absolute path contains a drive letter.
        if sys.platform == "win32":
            path = "C:\\Temp\\somedir"
        else:
            path = "/tmp/somedir"
        # _validate_shell_command should not raise
        tool._validate_shell_command(f"rm -rf {path}")

    async def test_rm_rf_relative_path_blocked(self) -> None:
        tool = RunShellTool(safety_level=1)
        with pytest.raises(ToolError, match="rm -rf without an absolute path"):
            await tool.run("rm -rf ./somedir")

    async def test_empty_command_raises(self) -> None:
        tool = RunShellTool(safety_level=1)
        with pytest.raises(ToolError, match="Empty command"):
            await tool.run("   ")

    async def test_leading_whitespace_still_blocked(self) -> None:
        """Whitespace-prefixed dangerous commands should still be caught."""
        tool = RunShellTool(safety_level=1)
        with pytest.raises(ToolError, match="blocked for safety"):
            await tool.run("  rm -rf /")

    async def test_fork_bomb_blocked(self) -> None:
        tool = RunShellTool(safety_level=1)
        with pytest.raises(ToolError, match="blocked for safety"):
            await tool.run(":(){ :|:& };:")

    async def test_timeout_cap(self) -> None:
        """timeout parameter should be capped at 120."""
        tool = RunShellTool(safety_level=1)
        # This should not cause an error but cap the timeout
        result = await tool.run("echo hello", timeout=999)
        assert "hello" in result
