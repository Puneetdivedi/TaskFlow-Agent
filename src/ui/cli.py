"""Rich terminal UI for the TaskFlow agent."""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from datetime import datetime

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.table import Table

from src.agent.orchestrator import AgentOrchestrator
from src.interfaces import IMemory, ISessionStore, ITaskStore
from src.interfaces.task_store import Task

console = Console()

BANNER = """
╔══════════════════════════════════════════════╗
║         🤖  TaskFlow Agent  v0.1             ║
║   Your autonomous day-to-day task assistant  ║
╚══════════════════════════════════════════════╝
"""

SESSION_USAGE = "Usage: /session <new|save|load|delete> [name]"

TASK_USAGE = (
    "Usage: /task <create <title> | list [status] | get <id> | "
    "update <id> <field=value> ... | complete <id> | delete <id>>\n"
    "update fields: title, description, status, priority, "
    "due_at (ISO-8601, e.g. 2026-08-10), every_days (0 = not recurring)"
)


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
) -> SessionCommandResult:
    """Handle a ``/session ...`` command.

    *command* is the text typed after ``/session`` (may be empty). Switching
    commands (``new``/``load``) checkpoint the current conversation to disk
    before switching, so nothing is lost.
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
        # Checkpoint the current conversation before switching.
        if current is not None and memory.messages:
            store.save(current, memory.messages)
        session_name = name or _default_session_name()
        if store.exists(session_name):
            return SessionCommandResult(
                current, f"Session '{session_name}' already exists — use /session load to open it."
            )
        try:
            # Saving the empty session validates the name early and makes
            # the new (empty) session visible in /sessions.
            store.save(session_name, [])
        except ValueError as exc:
            return SessionCommandResult(current, str(exc))
        memory.clear()
        return SessionCommandResult(session_name, f"Started new session '{session_name}'.")

    if sub == "save":
        if not memory.messages:
            return SessionCommandResult(current, "Nothing to save — conversation is empty.")
        target = name or current or _default_session_name()
        try:
            store.save(target, memory.messages)
        except ValueError as exc:
            return SessionCommandResult(current, str(exc))
        return SessionCommandResult(
            current, f"Saved session '{target}' ({len(memory.messages)} message(s))."
        )

    if sub == "load":
        if not name:
            return SessionCommandResult(current, SESSION_USAGE)
        if current is not None and memory.messages:
            store.save(current, memory.messages)
        try:
            messages = store.load(name)
        except KeyError as exc:
            return SessionCommandResult(current, str(exc))
        memory.restore(messages)
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
    lines.append(f"  created: {task.created_at[:19]}")
    return "\n".join(lines)


def format_task_list(tasks: list[Task]) -> str:
    """Return a human-readable listing of tasks (newest first)."""
    if not tasks:
        return "No tasks."
    lines = [f"{len(tasks)} task(s):"]
    for task in tasks:
        due = f", due: {task.due_at[:10]}" if task.due_at else ""
        lines.append(
            f"  [{task.status:<11}] {task.id} {task.title} "
            f"(priority: {task.priority}, created: {task.created_at[:10]}{due})"
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
        try:
            task = store.update(
                task_id,
                title=changes.get("title"),
                description=changes.get("description"),
                status=changes.get("status"),
                priority=changes.get("priority"),
                due_at=changes.get("due_at"),
                every_days=every_days,
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
) -> None:
    """Persist the current conversation before the CLI exits."""
    if current is not None and memory.messages:
        store.save(current, memory.messages)
        console.print(f"[dim]Saved session '{current}' ({len(memory.messages)} message(s)).[/dim]")


# ---------------------------------------------------------------------------
async def run_cli(
    orchestrator: AgentOrchestrator,
    session_store: ISessionStore,
    task_store: ITaskStore | None = None,
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
            console.print(
                f"[bold yellow]Reminder:[/bold yellow] {task.title} due {task.due_at[:10]}"
            )
        try:
            user_input = Prompt.ask("[bold cyan]You[/bold cyan]")
        except (EOFError, KeyboardInterrupt):
            _save_session_on_exit(session_store, orchestrator.memory, current_session)
            console.print("\n[bold yellow]Goodbye![/bold yellow]")
            break

        if not user_input.strip():
            continue

        # --- Built-in commands ---
        cmd = user_input.strip().lower()

        if cmd in ("/exit", "/quit", "exit", "quit"):
            _save_session_on_exit(session_store, orchestrator.memory, current_session)
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

        if cmd == "/task" or cmd.startswith("/task "):
            if task_store is None:
                console.print("[yellow]Tasks are not available in this build.[/yellow]")
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
            )
            if result.session is not None:
                current_session = result.session
            console.print(result.message)
            continue

        # --- Normal agent interaction ---
        with console.status("[bold yellow]Thinking...[/bold yellow]", spinner="dots"):
            try:
                response = await orchestrator.run(user_input)
            except Exception as exc:
                response = f"⚠️  Error: {exc}"

        console.print()
        console.print(Rule(style="dim"))
        console.print(Markdown(response))
        console.print(Rule(style="dim"))
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

    asyncio.run(run_cli(app.orchestrator, app.session_store, app.task_store))


if __name__ == "__main__":
    main()
