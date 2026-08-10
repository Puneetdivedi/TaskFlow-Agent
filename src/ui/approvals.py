"""Interactive tool-approval policy for the CLI.

A small pure policy (``approval_reason``) decides whether a call needs a
human gate, and ``make_approver`` turns that policy into an async
``ToolApprover`` that pauses the rich ``Live`` panel and prompts the user
for ``y``/``n``/``auto``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.prompt import Prompt

from src.interfaces.approval import ToolApprover

# Tools that always require explicit approval, whatever the arguments.
ALWAYS_APPROVE: frozenset[str] = frozenset({"delete_file", "move_file", "run_shell", "subagent"})

# File-writing tools approved only when they would overwrite an existing file.
OVERWRITE_CHECK: frozenset[str] = frozenset({"write_file", "yaml_write", "json_write"})

_PROMPT_TEXT = "[y]es / [n]o / [a]uto for rest of turn"


def approval_reason(name: str, args: dict[str, Any]) -> str | None:
    """Return a reason the call needs approval, or ``None`` to allow it."""
    if name in ALWAYS_APPROVE:
        return f"{name} can modify or execute outside the agent"
    if name in OVERWRITE_CHECK:
        path = args.get("path")
        if isinstance(path, str) and Path(path).expanduser().resolve().exists():
            return f"{name} would overwrite an existing file"
    return None


def make_approver(
    console: Console,
    *,
    enabled: bool = True,
    live: Live | None = None,
) -> ToolApprover | None:
    """Build a ``ToolApprover``, or ``None`` when approvals are disabled.

    ``live`` is the rich ``Live`` panel wrapping the turn; it is stopped for
    the duration of the prompt and restarted afterwards so the two don't
    fight over the terminal. When ``live`` is ``None`` (tests) the prompt is
    shown directly.

    Answers: ``y``/``yes`` allows the call; ``a``/``auto`` allows this and
    every remaining call in the turn; anything else (including an
    interrupted prompt) denies it.
    """
    if not enabled:
        return None

    auto_all = False

    async def approver(name: str, args: dict[str, Any]) -> str | None:
        nonlocal auto_all
        if auto_all:
            return None
        reason = approval_reason(name, args)
        if reason is None:
            return None
        if live is not None:
            live.stop()
        try:
            console.print(f"[bold yellow]⚠️  Approval needed:[/bold yellow] {name}")
            console.print(f"[dim]{reason}[/dim]")
            compact = json.dumps(args, ensure_ascii=False)[:200]
            console.print(f"[dim]args: {compact}[/dim]")
            answer = Prompt.ask(_PROMPT_TEXT, default="n")
        except EOFError:
            return reason  # fail-safe: an interrupted prompt denies the call
        finally:
            if live is not None:
                live.start()
        low = answer.strip().lower()
        if low in ("y", "yes"):
            return None
        if low in ("a", "auto"):
            auto_all = True
            return None
        return reason

    return approver
