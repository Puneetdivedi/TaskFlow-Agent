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
| `web_search` | Search the web via DuckDuckGo; returns numbered titles, URLs, and snippets (no API key) |
| `web_fetch` | Fetch an http(s) page and return its readable text (scripts/styles/tags stripped) |
| `yaml_read` | Read a YAML file and return its contents as normalized, canonical YAML |
| `yaml_write` | Write structured YAML to a file (parses/normalizes content, creates parent dirs) |
| `tasks` | Manage a persistent task list (create/list/get/update/complete/due/delete, due dates + recurrence) |

## MCP Server

TaskFlow's tools can be exposed to any Model Context Protocol client (Claude
Desktop, Claude Code, or a third-party host) as an MCP server. The server
advertises the same tools as the interactive agent — `read_file`, `run_shell`,
`web_search`, `yaml_read`, `tasks`, and friends — with identical names,
descriptions, and input schemas.

Start the server:

```bash
python -m src.mcp
```

It speaks MCP over **stdio**, so no ports or configuration are needed. In
Claude Desktop, point a new server entry at it:

```json
{
  "mcpServers": {
    "taskflow": {
      "command": "python",
      "args": ["-m", "src.mcp"]
    }
  }
}
```

The server shares TaskFlow's SQLite database (`~/.taskflow/taskflow.db`), so
tasks managed through MCP are the same ones the CLI sees. Tool results (and
any tool error) are returned as MCP text results.

## MCP Client

TaskFlow can also call **out to** external MCP servers. Configure them with
the `MCP_SERVERS` environment variable — a JSON object mapping a server name
to its launch details:

```json
{
  "filesystem": {
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
  }
}
```

At startup each configured server is queried over stdio for its tool list,
and every remote tool becomes a first-class agent tool — same name,
description, and input schema the remote server advertises — alongside the
built-ins and plugins. Call one and the arguments are forwarded to the remote
server; its text result becomes the tool result.

Notes:

- A server that fails to connect is logged and skipped — a bad config never
  blocks startup (each server is given up to 30s to respond).
- External tools are invoked through a fresh connection per call, which is
  simple and thread-safe; the cost is process startup on each call.
- `MCP_SERVERS` is read at startup. Setting it to `{}` (or leaving it unset)
  disables the client.

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
│   ├── web_tools.py       # DuckDuckGo search + page fetch (keyless)
│   ├── yaml_tools.py      # YAML read/write tools
│   ├── middleware.py      # Logging/audit tool pipeline
│   ├── security.py        # Shell command safety checks
│   └── registry.py        # Central tool registry + dispatch
├── memory/
│   ├── __init__.py        # Exports: ConversationMemory, FileIndex, SessionStore, TaskStore
│   ├── conversation.py    # Token-aware conversation history with pruning
│   ├── sqlite_store.py    # Shared SQLite backend + table schemas
│   ├── migration.py       # One-time legacy JSON → SQLite migration
│   ├── file_index.py      # Persistent filesystem index (SQLite)
│   ├── session_store.py   # Named session persistence (SQLite)
│   └── task_store.py      # Persistent task list (SQLite)
├── interfaces/
│   ├── __init__.py        # Exports: all protocols + dataclasses
│   └── task_store.py      # ITaskStore protocol + Task dataclass
├── di/
│   ├── __init__.py
│   ├── container.py       # Minimal DI container
│   └── factories.py       # Production wiring + AppComponents
├── mcp/
│   ├── __init__.py        # Exports: TaskFlowMCPServer
│   ├── server.py          # MCP server exposing TaskFlow's tools over stdio
│   ├── client.py          # MCP client: tools from external MCP_SERVERS
│   └── __main__.py        # Entry point: python -m src.mcp
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
├── test_migration.py       # Tests for the legacy JSON → SQLite migration
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
- [x] SQLite persistence (tasks, sessions, and file index in one durable database)
- [x] Web tools (DuckDuckGo search + page fetch — no API key)
- [x] YAML tools (read/write structured YAML files)
- [x] MCP server (expose all tools to any Model Context Protocol client)
- [x] MCP client (call tools from external MCP servers, configured via env)
- [x] API error handling (rate limits, timeouts, server errors)
- [x] Safety level enforcement
- [x] Rich CLI interface (colored output, tables, markdown)
- [x] Streaming responses (text streams live in the CLI as the model generates) with a clean Ctrl-C abort
- [x] Semantic memory (old turns condensed into a rolling summary, injected as context, persisted per session)
- [x] Sub-agents (delegate focused, isolated tasks to specialized roles)

## Sessions

Conversation history survives restarts via named, persistent sessions. When you
start the agent it offers to **resume the most recent session**; the current
session is **auto-saved on exit** and whenever you switch sessions.

Sessions are stored in the shared SQLite database at `~/.taskflow/taskflow.db`
(override with `TASKFLOW_DB`). Session names are slugs (`letters`, `digits`,
`.`, `_`, `-`).

- `/session new [name]` — start a fresh session (current one is saved first)
- `/session save [name]` — checkpoint the current conversation
- `/session load <name>` — open a saved session (current one is saved first)
- `/session delete <name>` — permanently remove a saved session
- `/sessions` — list all saved sessions

## Semantic Memory

Long conversations are compacted instead of silently forgotten. Once a
conversation's estimated token count passes `MEMORY_SUMMARY_THRESHOLD`
(default `60000`), the agent condenses the older turns into a **rolling
summary** — names, dates, decisions, preferences, file paths, and numbers —
injects it back into the model's context as an extra system block, and drops
the summarized turns from the live history. Recent context always stays
inline, and an exchange is never split mid-way (tool results stay with the
assistant turn that produced them).

The summary survives session save/load and CLI restarts: `/session save`
persists it, `/session load` and the resume prompt restore it. Summarization
is off when no summarizer is wired or when the threshold is `0` — set
`MEMORY_SUMMARY_THRESHOLD=0` to disable.

## Tasks

Tasks persist across restarts in the shared SQLite database. Manage them
directly from the CLI, or ask the agent to manage them with the `tasks` tool.

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

Tasks, sessions, and the file index all live in the shared SQLite database at
`~/.taskflow/taskflow.db` (override with `TASKFLOW_DB`). On first run any
legacy JSON data (`tasks.json`, `sessions/`, `file_index.json`) is migrated
into the database automatically.

## Configuration

Set these in `.env` (copy from `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | Your Anthropic API key (required) |
| `ANTHROPIC_MODEL` | `claude-sonnet-5-20250611` | Model to use |
| `SAFETY_LEVEL` | `1` | Permission tier (0–3) |
| `AGENT_WORK_DIR` | `.` | Working directory for the agent |
| `TASKFLOW_DB` | `~/.taskflow/taskflow.db` | Shared SQLite database (tasks, sessions, file index) |
| `MCP_SERVERS` | — | JSON config of external MCP servers to call (see [MCP Client](#mcp-client)) |

Web tools (`web_search` / `web_fetch`) are always available and use baked-in
defaults (15s timeout); no configuration required.

## Safety

TaskFlow uses a tiered permission model, enforced by the shell tool and extensible to other tools:

| Level | Behavior |
|-------|----------|
| 0 | **Lockdown** — shell execution blocked entirely |
| 1 | **Balanced** — auto for reads, ask for destructive writes (default) |
| 2 | **Permissive** — auto for files, warn on shell |
| 3 | **Autonomous** — full autonomy with logging |

Shell commands with known dangerous patterns (`rm -rf /`, `mkfs`, `dd`, `shutdown`, etc.) are blocked at all levels.

## Plugins

Third-party packages can add tools to the agent by registering an entry point
under the `taskflow.tools` group. Discovery runs once at startup in
`src/plugins`, and every discovered tool is available to the agent alongside
the built-ins.

An entry point may resolve to a `Tool` subclass, a `Tool` instance, or a
zero-argument callable returning one. The result must expose `name`,
`description`, `input_schema`, and an async `run(**kwargs) -> str` method:

```python
# my_package/tools.py
from typing import Any

from src.tools.base import Tool


class EchoTool(Tool):
    """Echoes the given text back — a minimal plugin tool."""

    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "Echoes the given text back to the caller."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "Text to echo"}},
            "required": ["text"],
        }

    async def run(self, **kwargs: Any) -> str:
        return f"Echo: {kwargs.get('text', '')}"
```

Register it in your package's `pyproject.toml`:

```toml
[project.entry-points."taskflow.tools"]
echo = "my_package.tools:EchoTool"
```

Contract notes:

- A plugin that fails to load, is not Tool-shaped, or collides with an existing
  tool name is logged and skipped — one bad plugin never blocks startup.
- On a name collision, the built-in tool (or the first plugin registered) wins.

## Sub-agents

The main agent can delegate a self-contained subtask to a **sub-agent** — a
named, focused role that runs its own short, isolated tool loop. This lets a
broad request be tackled phase by phase (research → code → review) without
polluting the main conversation history or exposing the full tool set at once.

The agent invokes a sub-agent through the `subagent` tool with a `name` and a
self-contained `task`; the sub-agent runs to completion and returns a final
report string. Each delegation:

- starts from a **fresh context** (the sub-agent never sees the main thread's history),
- is limited to a **restricted tool set** (it can only see and run its own tools),
- is capped at a **step budget** so no delegation can loop forever, and
- is never offered the `subagent` tool itself, so sub-agents cannot spawn
  unbounded descendants.

Three built-in roles ship with the agent:

| Role | Focus | Tools |
| --- | --- | --- |
| `researcher` | gather and verify information, cite sources | `web_search`, `web_fetch`, `read_file`, `list_files`, `search_files` |
| `coder` | read, implement, and verify changes | file tools, `run_shell`, `yaml_read`, `yaml_write` |
| `reviewer` | find correctness/security/style issues, ranked by severity | `read_file`, `list_files`, `search_files`, `run_shell` |

Sub-agent tool calls dispatch through the same tool registry as the main agent,
so middleware (logging/audit), plugin tools, and MCP tools behave identically
inside a sub-agent loop.
