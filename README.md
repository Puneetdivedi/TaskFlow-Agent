# TaskFlow Agent

Autonomous AI agent for automating day-to-day tasks.

## Quick Start

```bash
# Install dependencies
pip install -e .

# Set your API key
echo "ANTHROPIC_API_KEY=sk-..." > .env

# Run the agent
python -m src.main
```

## Project Structure

```
src/
├── main.py           # Entry point
├── agent/
│   ├── orchestrator.py   # Main agent loop
│   └── claude_client.py  # LLM interaction
├── tools/
│   ├── base.py           # Tool base class
│   ├── file_tools.py     # File system tools
│   ├── shell_tools.py    # Shell execution
│   └── registry.py       # Tool registry
├── memory/
│   ├── conversation.py   # Conversation history
│   └── file_index.py     # File system index
└── ui/
    └── cli.py            # Rich terminal UI
```

## Phase 1 Features

- [x] Agent orchestrator with LLM tool-calling loop
- [x] File system tools (read, write, list, search)
- [x] Shell execution tool (sandboxed)
- [x] Conversation memory
- [x] Rich CLI interface

## Safety

TaskFlow uses a tiered permission model:

| Level | Behavior |
|-------|----------|
| 0 | Ask before every action |
| 1 | Auto for reads, ask for writes |
| 2 | Auto for files, ask for shell |
| 3 | Full autonomy with logging |

Default is **Level 1**.
