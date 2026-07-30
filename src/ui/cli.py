"""Rich terminal UI for the TaskFlow agent."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from src.agent.orchestrator import AgentOrchestrator

console = Console()

BANNER = """
╔══════════════════════════════════════════════╗
║         🤖  TaskFlow Agent  v0.1             ║
║   Your autonomous day-to-day task assistant  ║
╚══════════════════════════════════════════════╝
"""


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


async def run_cli(orchestrator: AgentOrchestrator) -> None:
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

    while True:
        try:
            user_input = Prompt.ask("[bold cyan]You[/bold cyan]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[bold yellow]Goodbye![/bold yellow]")
            break

        if not user_input.strip():
            continue

        # --- Built-in commands ---
        cmd = user_input.strip().lower()

        if cmd in ("/exit", "/quit", "exit", "quit"):
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

    from src.di.factories import create_production_orchestrator

    orchestrator = create_production_orchestrator(settings=settings)

    asyncio.run(run_cli(orchestrator))


if __name__ == "__main__":
    main()
