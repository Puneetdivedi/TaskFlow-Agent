"""Rich terminal UI for the TaskFlow agent."""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from src.agent.automation import AutomationRunner
from src.agent.orchestrator import AgentOrchestrator
from src.interfaces import IFactStore, IMemory, ISessionStore, ITaskStore, format_facts
from src.interfaces.task_store import Task
from src.interfaces.usage import Usage
from src.ui.approvals import make_approver

console = Console()

BANNER = """
╔══════════════════════════════════════════════╗
║         🤖  TaskFlow Agent  v0.1             ║
║   Your autonomous day-to-day task assistant  ║
╚══════════════════════════════════════════════╝
"""

SESSION_USAGE = "Usage: /session <new|save|load|delete> [name]"

TASK_USAGE = (
    "Usage: /task <create <title> | list [status] | get <id> | run <id> | "
    "update <id> <field=value> ... | complete <id> | delete <id>>\n"
    "update fields: title, description, status, priority, "
    "due_at (ISO-8601, e.g. 2026-08-10), every_days (0 = not recurring), "
    "plan (<multi-step instructions>), auto_run (true/false)"
)

MEMORY_USAGE = "Usage: /remember <fact>\n       /recall [query]\n       /forget <id>"


def print_banner() -> None:
    console.print(BANNER, style="bold cyan")


def print_help() -> None:
    table = Table(title="Commands", title_style="bold", border_style="blue")
    table.add_column("Command", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")
    table.add_row("/help", "Show this help")
    table.add_row("/tools", "List available tools")
    table.add_row("/clear", "Clear conversation history")
    table.add_row("/history", "Show conversation history count")
    table.add_row("/session", "Manage sessions: new/save/load/delete")
    table.add_row("/sessions", "List saved sessions")
    table.add_row("/task", "Manage tasks: create/list/get/update/complete/delete")
    table.add_row("/tasks", "List all tasks")
    table.add_row("/reminders", "Show tasks due or overdue")
    table.add_row("/usage", "Show cumulative token usage and estimated cost")
    table.add_row("/remember", "Store a durable fact in cross-session memory")
    table.add_row("/recall", "Search or list remembered facts")
    table.add_row("/forget", "Delete a remembered fact by id")
    table.add_row("/exit", "Exit the agent")
    console.print(table)


def print_tools(orchestrator: AgentOrchestrator) -> None:
    defs = orchestrator.tools.anthropic_tool_defs()
    table = Table(title="Available Tools", title_style="bold", border_style="green")
    table.add_column("Tool", style="cyan", no_wrap=True)
    table.add_column("Description")
    for t in defs:
        table.add_row(t["name"], t["description"])
    console.print(table)


# ---------------------------------------------------------------------------
# Session commands
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SessionCommandResult:
    """Outcome of handling a ``/session`` command.

    ``session`` is the new current-session name, or ``None`` if it is
    unchanged; ``message`` is the text to show the user.
    """

    session: str | None
    message: str


@dataclass(frozen=True)
class TaskCommandResult:
    """Outcome of handling a ``/task`` command."""

    message: str


def _default_session_name() -> str:
    """Generate a timestamp-based session name that passes validation."""
    return datetime.now().strftime("%Y-%m-%d-%H%M%S")


def format_session_list(store: ISessionStore) -> str:
    """Return a human-readable listing of saved sessions."""
    sessions = store.list()
    if not sessions:
        return "No saved sessions yet. Use /session save [name] to create one."
    lines = ["Saved sessions:"]
    for info in sessions:
        # updated_at is ISO-8601 with microseconds — trim to seconds for display.
        shown = info.updated_at[:19] if len(info.updated_at) > 19 else info.updated_at
        lines.append(f"  {info.name:<40} {info.message_count:>5} msgs  {shown}")
    return "\n".join(lines)


def handle_session_command(
    command: str,
    store: ISessionStore,
    memory: IMemory,
    current: str | None,
    usage: Usage | None = None,
) -> SessionCommandResult:
    """Handle a ``/session ...`` command.

    *command* is the text typed after ``/session`` (may be empty). Switching
    commands (``new``/``load``) checkpoint the current conversation to disk
    before switching, so nothing is lost. The cumulative *usage* is persisted
    with each save so a session's spend survives restarts.
    """
    parts = command.strip().split()
    if not parts:
        if current is None:
            return SessionCommandResult(
                None, "No active session. Use /session new [name] to start one."
            )
        return SessionCommandResult(
            current,
            f"Current session: {current} ({len(memory.messages)} message(s))",
        )

    sub = parts[0].lower()
    name = parts[1] if len(parts) > 1 else None

    if sub == "new":
        # Checkpoint the current conversation (and its rolling summary)
        # before switching.
        if current is not None and memory.messages:
            store.save(current, memory.messages, memory.summary, usage=usage)
        session_name = name or _default_session_name()
        if store.exists(session_name):
            return SessionCommandResult(
                current, f"Session '{session_name}' already exists — use /session load to open it."
            )
        try:
            # Saving the empty session validates the name early and makes
            # the new (empty) session visible in /sessions.
            store.save(session_name, [], usage=usage)
        except ValueError as exc:
            return SessionCommandResult(current, str(exc))
        memory.clear()
        memory.set_summary("")
        return SessionCommandResult(session_name, f"Started new session '{session_name}'.")

    if sub == "save":
        if not memory.messages:
            return SessionCommandResult(current, "Nothing to save — conversation is empty.")
        target = name or current or _default_session_name()
        try:
            store.save(target, memory.messages, memory.summary, usage=usage)
        except ValueError as exc:
            return SessionCommandResult(current, str(exc))
        return SessionCommandResult(
            current, f"Saved session '{target}' ({len(memory.messages)} message(s))."
        )

    if sub == "load":
        if not name:
            return SessionCommandResult(current, SESSION_USAGE)
        if current is not None and memory.messages:
            store.save(current, memory.messages, memory.summary, usage=usage)
        try:
            messages = store.load(name)
        except KeyError as exc:
            return SessionCommandResult(current, str(exc))
        memory.restore(messages)
        memory.set_summary(store.load_summary(name))
        return SessionCommandResult(name, f"Loaded session '{name}' ({len(messages)} message(s)).")

    if sub == "delete":
        if not name:
            return SessionCommandResult(current, SESSION_USAGE)
        try:
            store.delete(name)
        except KeyError as exc:
            return SessionCommandResult(current, str(exc))
        return SessionCommandResult(current, f"Deleted session '{name}'.")

    return SessionCommandResult(current, f"Unknown /session command '{sub}'.\n{SESSION_USAGE}")


# ---------------------------------------------------------------------------
# Task commands
# ---------------------------------------------------------------------------
def format_task(task: Task) -> str:
    """Return a human-readable single-task description."""
    lines = [f"{task.id}: {task.title} [status: {task.status}, priority: {task.priority}]"]
    if task.description:
        lines.append(f"  {task.description}")
    if task.due_at:
        lines.append(f"  due: {task.due_at[:10]}")
    if task.every_days:
        lines.append(f"  repeats every {task.every_days} day(s)")
    if task.auto_run:
        lines.append("  ⚙  auto-runs when due")
    if task.plan:
        lines.append(f"  plan: {task.plan}")
    lines.append(f"  created: {task.created_at[:19]}")
    return "\n".join(lines)


def format_task_list(tasks: list[Task]) -> str:
    """Return a human-readable listing of tasks (newest first)."""
    if not tasks:
        return "No tasks."
    lines = [f"{len(tasks)} task(s):"]
    for task in tasks:
        due = f", due: {task.due_at[:10]}" if task.due_at else ""
        auto = ", auto-run" if task.auto_run else ""
        lines.append(
            f"  [{task.status:<11}] {task.id} {task.title} "
            f"(priority: {task.priority}, created: {task.created_at[:10]}{due}{auto})"
        )
    return "\n".join(lines)


def format_reminder_banner(tasks: list[Task]) -> str:
    """Return banner text listing tasks that are due or overdue."""
    return f"You have {len(tasks)} task(s) due or overdue:\n{format_task_list(tasks)}"


def newly_due_tasks(task_store: ITaskStore | None, reported: set[tuple[str, str]]) -> list[Task]:
    """Return due tasks not yet reported; record them so they don't repeat.

    Keyed on ``(task_id, due_at)`` so a recurring task whose due date rolls
    forward re-notifies on its next cycle.
    """
    if task_store is None:
        return []
    fresh = [t for t in task_store.due() if (t.id, t.due_at) not in reported]
    for t in fresh:
        reported.add((t.id, t.due_at))
    return fresh


def _parse_update_tokens(tokens: list[str]) -> dict[str, str] | None:
    """Parse ``field=value`` tokens, appending bare words to the previous value.

    Returns ``None`` when the token list is malformed (a bare word appears
    before any ``field=`` assignment).
    """
    changes: dict[str, str] = {}
    last_field: str | None = None
    for token in tokens:
        if "=" in token:
            field, _, value = token.partition("=")
            changes[field] = value
            last_field = field
        elif last_field is not None:
            changes[last_field] += f" {token}"
        else:
            return None
    return changes


def handle_task_command(command: str, store: ITaskStore) -> TaskCommandResult:
    """Handle a ``/task ...`` command.

    *command* is the text typed after ``/task`` (may be empty). Validation
    is delegated to the store; ``KeyError``/``ValueError`` messages are
    surfaced to the user unchanged.
    """
    parts = command.strip().split()
    if not parts or parts[0].lower() in ("help", "usage"):
        return TaskCommandResult(TASK_USAGE)

    sub = parts[0].lower()

    if sub == "create":
        title = " ".join(parts[1:]).strip()
        if not title:
            return TaskCommandResult(f"Task title required.\n{TASK_USAGE}")
        try:
            task = store.create(title)
        except ValueError as exc:
            return TaskCommandResult(str(exc))
        return TaskCommandResult(
            f"Created task {task.id}: {task.title} "
            f"(status: {task.status}, priority: {task.priority})"
        )

    if sub == "list":
        status = parts[1] if len(parts) > 1 else None
        try:
            tasks = store.list(status)
        except ValueError as exc:
            return TaskCommandResult(str(exc))
        return TaskCommandResult(format_task_list(tasks))

    if sub == "get":
        if len(parts) < 2:
            return TaskCommandResult(TASK_USAGE)
        try:
            task = store.get(parts[1])
        except KeyError as exc:
            return TaskCommandResult(str(exc))
        return TaskCommandResult(format_task(task))

    if sub == "update":
        if len(parts) < 3:
            return TaskCommandResult(TASK_USAGE)
        task_id = parts[1]
        changes = _parse_update_tokens(parts[2:])
        if changes is None:
            return TaskCommandResult(TASK_USAGE)
        unknown = set(changes) - {
            "title",
            "description",
            "status",
            "priority",
            "due_at",
            "every_days",
            "plan",
            "auto_run",
        }
        if unknown:
            fields = ", ".join(sorted(unknown))
            return TaskCommandResult(f"Unknown update field(s): {fields}.\n{TASK_USAGE}")
        every_days: int | None = None
        if "every_days" in changes:
            try:
                every_days = int(changes["every_days"])
            except ValueError:
                return TaskCommandResult(
                    f"Invalid every_days {changes['every_days']!r} — must be "
                    f"a non-negative integer.\n{TASK_USAGE}"
                )
        auto_run: bool | None = None
        if "auto_run" in changes:
            raw = changes["auto_run"].strip().lower()
            if raw in ("1", "true", "yes", "on"):
                auto_run = True
            elif raw in ("0", "false", "no", "off"):
                auto_run = False
            else:
                return TaskCommandResult(
                    f"Invalid auto_run {changes['auto_run']!r} — must be true/false.\n{TASK_USAGE}"
                )
        try:
            task = store.update(
                task_id,
                title=changes.get("title"),
                description=changes.get("description"),
                status=changes.get("status"),
                priority=changes.get("priority"),
                due_at=changes.get("due_at"),
                every_days=every_days,
                plan=changes.get("plan"),
                auto_run=auto_run,
            )
        except (KeyError, ValueError) as exc:
            return TaskCommandResult(str(exc))
        return TaskCommandResult(
            f"Updated task {task.id}: {task.title} "
            f"(status: {task.status}, priority: {task.priority})"
        )

    if sub == "complete":
        if len(parts) < 2:
            return TaskCommandResult(TASK_USAGE)
        try:
            task = store.complete(parts[1])
        except (KeyError, ValueError) as exc:
            return TaskCommandResult(str(exc))
        if task.every_days > 0 and task.due_at:
            return TaskCommandResult(
                f"Completed task {task.id}: {task.title} (next due: {task.due_at[:10]})"
            )
        return TaskCommandResult(f"Completed task {task.id}: {task.title}")

    if sub == "delete":
        if len(parts) < 2:
            return TaskCommandResult(TASK_USAGE)
        try:
            store.delete(parts[1])
        except KeyError as exc:
            return TaskCommandResult(str(exc))
        return TaskCommandResult(f"Deleted task {parts[1]}")

    return TaskCommandResult(f"Unknown /task command '{sub}'.\n{TASK_USAGE}")


def _save_session_on_exit(
    store: ISessionStore,
    memory: IMemory,
    current: str | None,
    usage: Usage | None = None,
) -> None:
    """Persist the current conversation (and its summary) before the CLI exits."""
    if current is not None and memory.messages:
        store.save(current, memory.messages, memory.summary, usage=usage)
        console.print(f"[dim]Saved session '{current}' ({len(memory.messages)} message(s)).[/dim]")


# ---------------------------------------------------------------------------
# Memory commands
# ---------------------------------------------------------------------------
def handle_remember_command(command: str, store: IFactStore) -> str:
    """Handle a ``/remember ...`` command (*command* is text after the prefix)."""
    content = command.strip()
    if not content:
        return f"Nothing to remember.\n{MEMORY_USAGE}"
    try:
        fact = store.add(content)
    except ValueError as exc:
        return str(exc)
    return f"Remembered fact {fact.id}: {fact.content}"


def handle_recall_command(command: str, store: IFactStore) -> str:
    """Handle a ``/recall ...`` command (*command* is text after the prefix).

    A query searches content/topic; without one the 10 most recent facts
    are listed.
    """
    query = command.strip()
    try:
        facts = store.search(query) if query else store.list(limit=10)
    except ValueError as exc:
        return str(exc)
    return format_facts(facts)


def handle_forget_command(command: str, store: IFactStore) -> str:
    """Handle a ``/forget ...`` command (*command* is text after the prefix)."""
    fact_id = command.strip()
    if not fact_id:
        return f"Memory id required.\n{MEMORY_USAGE}"
    try:
        store.delete(fact_id)
    except KeyError as exc:
        return str(exc)
    return f"Forgot memory {fact_id}."


# ---------------------------------------------------------------------------
# Usage display
# ---------------------------------------------------------------------------
def _format_tokens(n: int) -> str:
    """Format a token count compactly, e.g. ``12340`` -> ``"12.3k"``."""
    if n >= 100_000:
        return f"{n / 1000:.0f}k"
    if n >= 1_000:
        return f"{n / 1000:.1f}k"
    return str(n)


def format_usage_summary(usage: Usage, cost: float) -> str:
    """Return a compact one-line usage/cost summary (``""`` when idle).

    The line counts billed input (fresh plus cached), cached reads/writes,
    output, and the estimated cost — e.g.
    ``⚡ 12.3k in · 0.1k cached-read · 0.2k out · ~$0.0124``.
    """
    if usage.total_tokens == 0 and cost <= 0:
        return ""
    parts = [f"⚡ {_format_tokens(usage.total_input_tokens)} in"]
    if usage.cache_read_input_tokens:
        parts.append(f"{_format_tokens(usage.cache_read_input_tokens)} cached-read")
    if usage.cache_creation_input_tokens:
        parts.append(f"{_format_tokens(usage.cache_creation_input_tokens)} cached-write")
    if usage.output_tokens:
        parts.append(f"{_format_tokens(usage.output_tokens)} out")
    if cost > 0:
        parts.append(f"~${cost:.4f}")
    return " · ".join(parts)


# ---------------------------------------------------------------------------
# Streaming output
# ---------------------------------------------------------------------------
def _stream_renderable(text: str, tool_lines: list[str]) -> RenderableType:
    """Build the live panel: streamed markdown plus dim tool-call lines."""
    body: RenderableType
    if text.strip():
        body = Markdown(text)
    else:
        body = Text("")
    return Group(body, *(Text(f"  🔧 {line}", style="dim") for line in tool_lines))


def _stream_tool_label(name: str, args: dict[str, Any]) -> str:
    """Render a compact tool-call label like ``run_shell(cmd="ls -la")``.

    Argument values are truncated so a long value never blows up the panel.
    """
    parts = [f"{k}={_truncate(str(v))}" for k, v in args.items()]
    return f"{name}({', '.join(parts)})"


def _truncate(value: str, limit: int = 80) -> str:
    """Truncate *value* to *limit* characters, appending an ellipsis."""
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _should_print_final(streamed: str, response: str) -> bool:
    """Whether the final response must be printed after the streaming panel.

    The answer is already on screen when it was streamed as the tail of the
    panel; it must be printed when it wasn't — error returns, stop-sequence
    markers, and max-tool-calls notices are never streamed.
    """
    return bool(response.strip()) and not streamed.rstrip().endswith(response.strip())


async def _run_streamed_turn(
    orchestrator: AgentOrchestrator,
    user_input: str,
    console: Console,
    *,
    approvals_enabled: bool = False,
) -> tuple[str, str]:
    """Run one turn while streaming the assistant's text live.

    Returns ``(final_response_text, streamed_text)``. *streamed_text* is
    what appeared in the live panel; *final_response_text* is whatever
    ``run`` returned, which may differ (error returns, stop markers). When
    *approvals_enabled* is set, risky tool calls pause the panel and ask
    the user to confirm.
    """
    text_buffer: list[str] = []
    tool_lines: list[str] = []

    with Live(
        _stream_renderable("", []),
        console=console,
        refresh_per_second=12,
        vertical_overflow="visible",
    ) as live:

        async def _on_delta(delta: str) -> None:
            text_buffer.append(delta)
            live.update(_stream_renderable("".join(text_buffer), tool_lines))

        async def _on_tool(name: str, args: dict[str, Any]) -> None:
            tool_lines.append(_stream_tool_label(name, args))
            live.update(_stream_renderable("".join(text_buffer), tool_lines))

        approver = make_approver(console, enabled=approvals_enabled, live=live)

        response = await orchestrator.run(
            user_input,
            on_text_delta=_on_delta,
            on_tool_call=_on_tool,
            tool_approver=approver,
        )

    return response, "".join(text_buffer)


# ---------------------------------------------------------------------------
async def run_cli(
    orchestrator: AgentOrchestrator,
    session_store: ISessionStore,
    task_store: ITaskStore | None = None,
    fact_store: IFactStore | None = None,
    automation: AutomationRunner | None = None,
    *,
    approvals_enabled: bool = False,
) -> None:
    """Main interactive loop."""
    print_banner()
    console.print(
        Panel(
            "I'm your AI task assistant. Tell me what you'd like me to do.\n"
            "Try: 'list my files', 'read README.md', or 'search for TODO'.\n"
            "Type [bold]/help[/bold] for all commands.",
            border_style="blue",
        )
    )
    console.print()

    current_session: str | None = None

    # Offer to resume the most recently saved session.
    latest = session_store.latest()
    if latest:
        answer = Prompt.ask(f"Resume last session '{latest}'? [y/N]", default="n")
        if answer.strip().lower() in ("y", "yes"):
            messages = session_store.load(latest)
            orchestrator.memory.restore(messages)
            orchestrator.memory.set_summary(session_store.load_summary(latest))
            current_session = latest
            console.print(f"[green]Resumed session '{latest}'.[/green]")

    # Track (task_id, due_at) pairs already surfaced so reminders don't repeat.
    reported: set[tuple[str, str]] = set()
    if task_store is not None:
        startup_due = task_store.due()
        if startup_due:
            console.print(
                Panel(
                    format_reminder_banner(startup_due),
                    title="Reminders",
                    border_style="yellow",
                )
            )
            reported.update((t.id, t.due_at) for t in startup_due)

    while True:
        for task in newly_due_tasks(task_store, reported):
            if task.auto_run and task.plan and automation is not None:
                console.print(
                    f"[bold yellow]Auto-running:[/bold yellow] {task.title} "
                    f"due {task.due_at[:10]}"
                )
                report = await automation.run_plan(task.plan, title=task.title)
                try:
                    task_store.advance(task.id)
                except KeyError:
                    pass
                console.print(
                    Panel(report, title=f"Automated: {task.title}", border_style="green")
                )
            else:
                console.print(
                    f"[bold yellow]Reminder:[/bold yellow] {task.title} due {task.due_at[:10]}"
                )
        try:
            user_input = Prompt.ask("[bold cyan]You[/bold cyan]")
        except (EOFError, KeyboardInterrupt):
            _save_session_on_exit(
                session_store,
                orchestrator.memory,
                current_session,
                usage=orchestrator.usage,
            )
            console.print("\n[bold yellow]Goodbye![/bold yellow]")
            break

        if not user_input.strip():
            continue

        # --- Built-in commands ---
        cmd = user_input.strip().lower()

        if cmd in ("/exit", "/quit", "exit", "quit"):
            _save_session_on_exit(
                session_store,
                orchestrator.memory,
                current_session,
                usage=orchestrator.usage,
            )
            console.print("[bold yellow]Goodbye![/bold yellow]")
            break

        if cmd == "/help":
            print_help()
            continue

        if cmd == "/tools":
            print_tools(orchestrator)
            continue

        if cmd == "/clear":
            orchestrator.memory.clear()
            console.print("[green]Conversation cleared.[/green]")
            continue

        if cmd == "/history":
            count = len(orchestrator.memory.messages)
            console.print(f"[blue]Conversation has {count} message(s).[/blue]")
            continue

        if cmd == "/sessions":
            console.print(format_session_list(session_store))
            continue

        if cmd == "/tasks":
            if task_store is None:
                console.print("[yellow]Tasks are not available in this build.[/yellow]")
                continue
            console.print(format_task_list(task_store.list()))
            continue

        if cmd == "/reminders":
            if task_store is None:
                console.print("[yellow]Tasks are not available in this build.[/yellow]")
                continue
            console.print(format_task_list(task_store.due()))
            continue

        if cmd == "/usage":
            usage_line = format_usage_summary(orchestrator.usage, orchestrator.estimated_cost())
            console.print(usage_line if usage_line else "[dim]No LLM usage yet.[/dim]")
            continue

        if cmd == "/task" or cmd.startswith("/task "):
            if task_store is None:
                console.print("[yellow]Tasks are not available in this build.[/yellow]")
                continue
            sub = user_input[len("/task") :].strip().split()
            if sub and sub[0].lower() == "run":
                if automation is None:
                    console.print("[yellow]Automation is not available in this build.[/yellow]")
                    continue
                if len(sub) < 2:
                    console.print(TASK_USAGE)
                    continue
                try:
                    task = task_store.get(sub[1])
                except KeyError as exc:
                    console.print(str(exc))
                    continue
                if not task.plan:
                    console.print(
                        f"[yellow]Task {task.id} has no plan — set one with "
                        f"/task update {task.id} plan=<steps>[/yellow]"
                    )
                    continue
                console.print(f"[bold]Running plan for {task.id}: {task.title}…[/bold]")
                report = await automation.run_plan(task.plan, title=task.title)
                try:
                    task_store.advance(task.id)
                except KeyError:
                    pass
                console.print(Panel(report, title=f"Automated: {task.title}", border_style="green"))
                continue
            task_result = handle_task_command(user_input[len("/task") :], task_store)
            console.print(task_result.message)
            continue

        if cmd == "/session" or cmd.startswith("/session "):
            result = handle_session_command(
                user_input[len("/session") :],
                session_store,
                orchestrator.memory,
                current_session,
                usage=orchestrator.usage,
            )
            if result.session is not None:
                current_session = result.session
            console.print(result.message)
            continue

        if fact_store is not None:
            if cmd == "/remember" or cmd.startswith("/remember "):
                console.print(handle_remember_command(user_input[len("/remember") :], fact_store))
                continue

            if cmd == "/recall" or cmd.startswith("/recall "):
                console.print(handle_recall_command(user_input[len("/recall") :], fact_store))
                continue

            if cmd == "/forget" or cmd.startswith("/forget "):
                console.print(handle_forget_command(user_input[len("/forget") :], fact_store))
                continue
        elif cmd in ("/remember", "/recall", "/forget") or any(
            cmd.startswith(f"/{name} ") for name in ("remember", "recall", "forget")
        ):
            console.print("[yellow]Cross-session memory is not available in this build.[/yellow]")
            continue

        # --- Normal agent interaction ---
        before = orchestrator.usage
        cost_before = orchestrator.estimated_cost()
        try:
            response, streamed = await _run_streamed_turn(
                orchestrator,
                user_input,
                console,
                approvals_enabled=approvals_enabled,
            )
        except KeyboardInterrupt:
            console.print("\n[bold yellow]Turn aborted.[/bold yellow]")
            continue
        except Exception as exc:
            console.print(f"\n[red]⚠️  Error: {exc}[/red]")
            continue

        console.print(Rule(style="dim"))
        if _should_print_final(streamed, response):
            console.print(Markdown(response))
        usage_line = format_usage_summary(
            orchestrator.usage - before,
            orchestrator.estimated_cost() - cost_before,
        )
        if usage_line:
            console.print(f"[dim]{usage_line}[/dim]")
        console.print()


def main() -> None:
    """Entry point for the CLI application."""
    from src.logging_config import configure_logging

    configure_logging()

    from config.settings import init_settings

    settings = init_settings()

    if not settings.is_ready:
        console.print(
            "[bold red]ERROR:[/bold red] ANTHROPIC_API_KEY not set.\n"
            "Copy [bold].env.example[/bold] to [bold].env[/bold] and add your key."
        )
        sys.exit(1)

    from src.di.factories import create_production_app

    app = create_production_app(settings=settings)

    asyncio.run(
        run_cli(
            app.orchestrator,
            app.session_store,
            app.task_store,
            app.fact_store,
            app.automation,
            approvals_enabled=settings.tool_approvals_enabled,
        )
    )


if __name__ == "__main__":
    main()
