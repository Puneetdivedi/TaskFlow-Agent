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
│   └── registry.py        # Central tool registry + dispatch
├── memory/
│   ├── __init__.py        # Exports: ConversationMemory, FileIndex, SessionStore
│   ├── conversation.py    # Token-aware conversation history with pruning
│   ├── file_index.py      # Persistent filesystem index (JSON cache)
│   └── session_store.py   # Named session persistence (JSON)
└── ui/
    ├── __init__.py         # Exports: cli_main
    └── cli.py              # Rich terminal UI (panels, tables, markdown)
config/
├── __init__.py             # Exports: SETTINGS, Settings
└── settings.py             # Environment-driven config (API key, model, safety)
tests/
├── test_file_tools.py      # Tests for all file tools
└── test_orchestrator.py    # Tests for registry + orchestrator init
```

## Features

- [x] Agent orchestrator with Claude tool-calling loop
- [x] File system tools (read, write, list, search, move, delete)
- [x] File system index (cache and query directory structure)
- [x] Shell execution tool with safety restrictions
- [x] Conversation memory with token-aware pruning
- [x] Persistent named sessions (save/load/resume conversations)
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

## Configuration

Set these in `.env` (copy from `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | Your Anthropic API key (required) |
| `ANTHROPIC_MODEL` | `claude-sonnet-5-20250611` | Model to use |
| `SAFETY_LEVEL` | `1` | Permission tier (0–3) |
| `AGENT_WORK_DIR` | `.` | Working directory for the agent |
| `SESSION_DIR` | `~/.taskflow/sessions` | Where named sessions are stored |

## Safety

TaskFlow uses a tiered permission model, enforced by the shell tool and extensible to other tools:

| Level | Behavior |
|-------|----------|
| 0 | **Lockdown** — shell execution blocked entirely |
| 1 | **Balanced** — auto for reads, ask for destructive writes (default) |
| 2 | **Permissive** — auto for files, warn on shell |
| 3 | **Autonomous** — full autonomy with logging |

Shell commands with known dangerous patterns (`rm -rf /`, `mkfs`, `dd`, `shutdown`, etc.) are blocked at all levels.
