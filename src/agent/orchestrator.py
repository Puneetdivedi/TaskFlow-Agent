"""Main agent loop — decides *what* to do, calls tools, and returns answers."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from src.agent.claude_client import ClaudeClient
from src.memory.conversation import ConversationMemory
from src.tools.base import ToolError
from src.tools.registry import ToolRegistry

# Default system prompt that shapes the agent's personality and constraints.
DEFAULT_SYSTEM_PROMPT = """\
You are **TaskFlow Agent**, an autonomous AI assistant that helps users automate \
their day-to-day tasks.

## Your capabilities
You have access to tools that let you:
- Read, write, list, search, move, and delete files
- Execute shell commands (with safety restrictions)
- (More tools coming in future phases)

## How you work
1. **Understand** — Parse the user's request carefully.
2. **Plan** — Break it into steps. Use a chain of thought.
3. **Act** — Call tools one at a time. Wait for results before proceeding.
4. **Report** — Summarize what you did and what the result was.

## Constraints
- Always ask before doing something destructive (delete, overwrite critical files).
- If a tool fails, try an alternative approach or explain the limitation.
- If you're unsure, ask the user to clarify.
- Be concise but informative. Don't over-explain trivial steps.
"""


class AgentOrchestrator:
    """Drives the LLM tool-calling loop: prompt → tool call → result → next action."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-5-20250611",
        system_prompt: str | None = None,
        work_dir: Path | None = None,
        max_tool_calls: int = 25,
    ) -> None:
        self._client = ClaudeClient(api_key=api_key, model=model)
        self._tools = ToolRegistry(work_dir=work_dir)
        self._memory = ConversationMemory()
        self._system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self._max_tool_calls = max_tool_calls

    # ------------------------------------------------------------------
    @property
    def tools(self) -> ToolRegistry:
        return self._tools

    @property
    def memory(self) -> ConversationMemory:
        return self._memory

    # ------------------------------------------------------------------
    async def run(self, user_input: str) -> str:
        """Process a single user request through the full tool-calling loop.

        Returns the final assistant response string.
        """
        self._memory.add_user(user_input)

        tool_call_count = 0

        while True:
            response = self._client.send_messages(
                messages=self._memory.messages,
                system=self._system_prompt,
                tools=self._tools.anthropic_tool_defs(),
            )

            # --- Process each content block ---
            for block in response.content:
                if block.type == "text":
                    # Claude is speaking — we save it and continue
                    # (the loop may still have tool_use blocks to process)
                    continue

                if block.type == "tool_use":
                    tool_call_count += 1
                    if tool_call_count > self._max_tool_calls:
                        error_msg = (
                            f"Exceeded max tool calls ({self._max_tool_calls}). "
                            "Aborting to prevent runaway execution."
                        )
                        self._memory.add_assistant(
                            [
                                {"type": "text", "text": error_msg},
                                block,  # include the tool_use for consistency
                            ]
                        )
                        self._memory.add_tool_result(
                            block.id, error_msg
                        )
                        return error_msg

                    # Execute the tool
                    try:
                        result = await self._tools.dispatch(
                            block.name, block.input
                        )
                    except ToolError as exc:
                        result = f"Error: {exc}"

                    # Append assistant block + tool result to memory
                    self._memory.add_assistant(
                        [{"type": "text", "text": ""}, block]
                    )
                    self._memory.add_tool_result(block.id, result)

            # --- Decide whether to stop ---
            if response.stop_reason == "end_turn":
                # Extract final text response
                final_text = ""
                for block in response.content:
                    if block.type == "text":
                        final_text += block.text
                # Also capture any text from tool_use rounds
                # (the final text block is the summary)
                if not final_text:
                    final_text = "(No response text)"
                self._memory.add_assistant(final_text)
                return final_text

            if response.stop_reason == "stop_sequence":
                return "[Agent stopped — stop sequence encountered]"

            # Otherwise continue the loop (tool_use was the stop reason)
