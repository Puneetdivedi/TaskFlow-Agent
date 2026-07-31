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
from src.interfaces import IMemory, ISessionStore

console = Console()

BANNER = """
╔══════════════════════════════════════════════╗
║         🤖  TaskFlow Agent  v0.1             ║
║   Your autonomous day-to-day task assistant  ║
╚══════════════════════════════════════════════╝
"""

SESSION_USAGE = "Usage: /session <new|save|load|delete> [name]"


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
async def run_cli(orchestrator: AgentOrchestrator, session_store: ISessionStore) -> None:
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

    while True:
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

    asyncio.run(run_cli(app.orchestrator, app.session_store))


if __name__ == "__main__":
    main()
