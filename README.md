# TaskFlow Agent

Autonomous AI agent for automating day-to-day tasks, powered by Claude.

## Quick Start

```bash
# Install dependencies
pip install -e .

# Set your API key
echo "ANTHROPIC_API_KEY=sk-..." > .env

# Run the agent
python -m src.main
```

## CLI Commands

| Command | Action |
|---------|--------|
| `/help` | Show available commands |
| `/tools` | List all available agent tools |
| `/clear` | Clear conversation history |
| `/history` | Show message count |
| `/session new [name]` | Start a new session (auto-saves the current one) |
| `/session save [name]` | Save the current conversation to a named session |
| `/session load <name>` | Load a saved session (auto-saves the current one) |
| `/session delete <name>` | Delete a saved session |
| `/sessions` | List saved sessions |
| `/tasks` | List all tasks |
| `/task create <title>` | Create a new task |
| `/task list [status]` | List tasks, optionally filtered by status |
| `/task get <id>` | Show one task's details |
| `/task update <id> <field=value> ...` | Update a task's title/description/status/priority/due_at/every_days |
| `/task complete <id>` | Mark a task as done (recurring tasks roll to their next due date) |
| `/task delete <id>` | Delete a task |
| `/reminders` | Show tasks due or overdue |
| `/exit` | Exit the agent (auto-saves the current session) |

## Available Tools

The agent can use these tools to accomplish tasks autonomously:

| Tool | Description |
|------|-------------|
| `read_file` | Read contents of a text file |
| `write_file` | Write content to a file (creates parent dirs) |
| `list_files` | List directory contents with sizes and timestamps |
| `search_files` | Regex search inside files via ripgrep |
| `move_file` | Move or rename files/directories |
| `delete_file` | Delete files or empty directories |
| `file_index` | Cache and query filesystem layout for quick lookups |
| `run_shell` | Execute shell commands (safety-restricted) |
| `tasks` | Manage a persistent task list (create/list/get/update/complete/due/delete, due dates + recurrence) |

## Project Structure

```
src/
├── main.py                # Entry point
├── agent/
│   ├── __init__.py        # Exports: AgentOrchestrator, ClaudeClient
│   ├── orchestrator.py    # Main agent loop with tool-calling logic
│   └── claude_client.py   # Anthropic SDK wrapper with error handling
├── tools/
│   ├── __init__.py        # Exports: all tools + ToolRegistry
│   ├── base.py            # Tool ABC and ToolError
│   ├── file_tools.py      # Read, Write, List, Search, Move, Delete, FileIndex
│   ├── shell_tools.py     # RunShell with safety guards
│   ├── task_tools.py      # Persistent task-management tool
│   ├── middleware.py      # Logging/audit tool pipeline
│   ├── security.py        # Shell command safety checks
│   └── registry.py        # Central tool registry + dispatch
├── memory/
│   ├── __init__.py        # Exports: ConversationMemory, FileIndex, SessionStore, TaskStore
│   ├── conversation.py    # Token-aware conversation history with pruning
│   ├── file_index.py      # Persistent filesystem index (JSON cache)
│   ├── session_store.py   # Named session persistence (JSON)
│   └── task_store.py      # Persistent task list (JSON)
├── interfaces/
│   ├── __init__.py        # Exports: all protocols + dataclasses
│   └── task_store.py      # ITaskStore protocol + Task dataclass
├── di/
│   ├── __init__.py
│   ├── container.py       # Minimal DI container
│   └── factories.py       # Production wiring + AppComponents
├── plugins/
│   └── __init__.py        # Entry-point tool discovery
└── ui/
    ├── __init__.py         # Exports: cli_main
    └── cli.py              # Rich terminal UI (panels, tables, markdown)
config/
├── __init__.py             # Exports: SETTINGS, Settings
└── settings.py             # Environment-driven config (API key, model, safety)
tests/
├── test_cli.py             # Tests for CLI helpers + command handlers
├── test_task_store.py      # Tests for the persistent task store
├── test_task_tools.py      # Tests for the tasks tool
└── ...                     # Plus unit tests for agent, tools, sessions, DI, etc.
```

## Features

- [x] Agent orchestrator with Claude tool-calling loop
- [x] File system tools (read, write, list, search, move, delete)
- [x] File system index (cache and query directory structure)
- [x] Shell execution tool with safety restrictions
- [x] Conversation memory with token-aware pruning
- [x] Persistent named sessions (save/load/resume conversations)
- [x] Persistent task list (create/list/get/update/complete/delete)
- [x] Task scheduling (due dates + recurrence) and reminders
- [x] API error handling (rate limits, timeouts, server errors)
- [x] Safety level enforcement
- [x] Rich CLI interface (colored output, tables, markdown)

## Sessions

Conversation history survives restarts via named, persistent sessions. When you
start the agent it offers to **resume the most recent session**; the current
session is **auto-saved on exit** and whenever you switch sessions.

Sessions are stored as JSON files under `~/.taskflow/sessions/` (override with
`SESSION_DIR`). Session names are slugs (`letters`, `digits`, `.`, `_`, `-`) so
they are safe to use as filenames.

- `/session new [name]` — start a fresh session (current one is saved first)
- `/session save [name]` — checkpoint the current conversation
- `/session load <name>` — open a saved session (current one is saved first)
- `/session delete <name>` — permanently remove a saved session
- `/sessions` — list all saved sessions

## Tasks

Tasks persist across restarts in a single JSON file. Manage them directly from
the CLI, or ask the agent to manage them with the `tasks` tool.

- `/tasks` — list all tasks (newest first)
- `/task create <title>` — create a task
- `/task list [status]` — list tasks, optionally filtered by status (`todo`, `in_progress`, `done`)
- `/task get <id>` — show one task's details
- `/task update <id> <field=value> ...` — update `title`, `description`, `status`, `priority` (`low`, `medium`, `high`), `due_at`, or `every_days`
- `/task complete <id>` — mark a task done
- `/task delete <id>` — remove a task
- `/reminders` — show tasks due or overdue

Scheduling:

- `/task update <id> due_at=2026-08-10` — set a due date (ISO-8601; `due_at=` clears it)
- `/task update <id> every_days=7` — make a task recur every 7 days (`every_days=0` clears)
- `/task complete <id>` — completing a recurring task rolls its due date forward by `every_days` and keeps it in `todo`; the response shows the next due date

The CLI shows a **Reminders** banner on startup for tasks already due, and ticks after each turn so a task that becomes due mid-session is surfaced without being asked. Recurring tasks re-notify on each new cycle.

Tasks are stored as JSON in `~/.taskflow/tasks.json` (override with `TASKS_FILE`).

## Configuration

Set these in `.env` (copy from `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | Your Anthropic API key (required) |
| `ANTHROPIC_MODEL` | `claude-sonnet-5-20250611` | Model to use |
| `SAFETY_LEVEL` | `1` | Permission tier (0–3) |
| `AGENT_WORK_DIR` | `.` | Working directory for the agent |
| `SESSION_DIR` | `~/.taskflow/sessions` | Where named sessions are stored |
| `TASKS_FILE` | `~/.taskflow/tasks.json` | Where the task list is stored |

## Safety

TaskFlow uses a tiered permission model, enforced by the shell tool and extensible to other tools:

| Level | Behavior |
|-------|----------|
| 0 | **Lockdown** — shell execution blocked entirely |
| 1 | **Balanced** — auto for reads, ask for destructive writes (default) |
| 2 | **Permissive** — auto for files, warn on shell |
| 3 | **Autonomous** — full autonomy with logging |

Shell commands with known dangerous patterns (`rm -rf /`, `mkfs`, `dd`, `shutdown`, etc.) are blocked at all levels.
