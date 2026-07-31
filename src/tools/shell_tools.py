"""Shell execution tool — runs commands with safety guards."""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
from pathlib import Path
from typing import Any

from src.tools.base import Tool, ToolError

logger = logging.getLogger(__name__)

# Commands that are NEVER allowed regardless of safety level
FORBIDDEN_PREFIXES = [
    "rm -rf /",
    "rm -rf /*",
    "format ",
    "mkfs",
    "dd if=",
    ":(){ :|:& };:",  # fork bomb
    "chmod 000",
    "chown ",
    "> /dev/",  # raw device
    "shutdown",
    "reboot",
    "halt",
]


class RunShellTool(Tool):
    """Execute a shell command and capture its output."""

    def __init__(
        self,
        work_dir: Path | None = None,
        safety_level: int = 1,
    ) -> None:
        self._work_dir = work_dir or Path.cwd()
        self._safety_level = safety_level

    @property
    def name(self) -> str:
        return "run_shell"

    @property
    def description(self) -> str:
        return (
            "Execute a shell command and return its stdout and stderr. "
            "The command runs in the project's working directory. "
            "Long-running commands will time out after 60 seconds."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (max 120, default 60)",
                },
                "work_dir": {
                    "type": "string",
                    "description": ("Working directory for the command. Defaults to project root."),
                },
            },
            "required": ["command"],
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _validate_shell_command(command: str) -> None:
        """Check *command* against safety rules; raise ``ToolError`` if blocked."""
        stripped = command.strip().lower()
        if not stripped:
            raise ToolError("Empty command.")

        # Forbidden-prefix check (e.g. ``rm -rf /``, ``dd if=``).
        for forbidden in FORBIDDEN_PREFIXES:
            if stripped.startswith(forbidden):
                raise ToolError(f"Command blocked for safety: {forbidden!r} is not allowed.")

        # Prevent ``rm -rf`` with a non-absolute target.
        try:
            tokens = shlex.split(command, posix=False)
        except ValueError as exc:
            raise ToolError(f"Cannot parse command: {exc}") from exc

        for i, token in enumerate(tokens):
            if token == "rm" and i + 2 < len(tokens):
                next_token = tokens[i + 1].replace("-", "").replace("r", "").replace("f", "")
                # If after stripping -r/f there's nothing (or just "rf"),
                # the next token is the target.
                if next_token in ("f", "rf", ""):
                    target = tokens[i + 2]
                    if not os.path.isabs(target):
                        raise ToolError(
                            "rm -rf without an absolute path is blocked — "
                            "please use the delete_file tool instead."
                        )

    async def run(  # type: ignore[override]
        self,
        command: str,
        timeout: int = 60,
        work_dir: str | None = None,
        **kwargs: Any,
    ) -> str:
        # --- Safety level enforcement ---
        if self._safety_level == 0:
            raise ToolError(
                "Shell execution is blocked at safety level 0. "
                "Set SAFETY_LEVEL=1 or higher in .env to allow shell commands."
            )

        # --- Validation ---
        self._validate_shell_command(command)

        cwd = Path(work_dir).resolve() if work_dir else self._work_dir
        timeout = min(timeout, 120)  # hard cap

        logger.warning("Shell execution: command=%.200r cwd=%s", command, cwd)

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(cwd),
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            return f"Command timed out after {timeout}s (output so far discarded)."
        except Exception as exc:
            raise ToolError(f"Shell execution failed: {exc}") from exc

        out_text = stdout.decode("utf-8", errors="replace").strip()
        err_text = stderr.decode("utf-8", errors="replace").strip()

        parts = [f"Exit code: {proc.returncode}"]
        if out_text:
            parts.append(f"\n[stdout]\n{out_text}")
        if err_text:
            parts.append(f"\n[stderr]\n{err_text}")

        return "\n".join(parts)
