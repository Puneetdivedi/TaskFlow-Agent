"""Main agent loop — decides *what* to do, calls tools, and returns answers."""

from __future__ import annotations

import logging
from typing import Any

from src.interfaces import IMemory, IToolRegistry, LLMClient
from src.tools.base import ToolError

logger = logging.getLogger(__name__)

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
    """Drives the LLM tool-calling loop: prompt → tool call → result → next action.

    All external dependencies (LLM client, tool registry, memory) are
    injected via the constructor — no concrete implementations are
    created here.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        tools: IToolRegistry,
        memory: IMemory,
        system_prompt: str | None = None,
        max_tool_calls: int = 25,
    ) -> None:
        self._client = llm_client
        self._tools = tools
        self._memory = memory
        self._system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self._max_tool_calls = max_tool_calls

    # ------------------------------------------------------------------
    @property
    def tools(self) -> IToolRegistry:
        return self._tools

    @property
    def memory(self) -> IMemory:
        return self._memory

    # ------------------------------------------------------------------
    async def run(self, user_input: str) -> str:
        """Process a single user request through the full tool-calling loop.

        Returns the final assistant response string.
        """
        self._memory.add_user(user_input)
        logger.info("Processing user input: %.200s", user_input)

        tool_call_count = 0

        while True:
            try:
                response = await self._client.send_messages(
                    messages=self._memory.messages,
                    system=self._system_prompt,
                    tools=self._tools.anthropic_tool_defs(),
                )
            except Exception as exc:  # broad catch — interface impls may raise different errors
                error_text = f"API error: {exc}"
                logger.error("API call failed: %s", exc)
                self._memory.add_assistant(error_text)
                return error_text

            # --- Build a single assistant response from all content blocks ---
            assistant_content: list[dict] = []
            tool_blocks: list[Any] = []

            for block in response.content:
                if block.type == "text":
                    assistant_content.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    tool_blocks.append(block)
                    assistant_content.append(
                        {
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        }
                    )

            # Preserve empty assistant entries so the message alternation
            # (user → assistant → (tool_result) → assistant → …) stays valid.
            if assistant_content or tool_blocks:
                self._memory.add_assistant(assistant_content if assistant_content else "")

            # --- Execute each tool call ---
            for block in tool_blocks:
                tool_call_count += 1
                logger.info("Dispatching tool: %s with args: %s", block.name, block.input)
                if tool_call_count > self._max_tool_calls:
                    error_msg = (
                        f"Exceeded max tool calls ({self._max_tool_calls}). "
                        "Aborting to prevent runaway execution."
                    )
                    logger.warning("Max tool calls exceeded: %d", self._max_tool_calls)
                    self._memory.add_tool_result(block.id, error_msg)
                    return error_msg

                try:
                    result = await self._tools.dispatch(block.name, block.input)
                except ToolError as exc:
                    result = f"Error: {exc}"

                logger.debug("Tool result for %s: %.500s", block.name, result)
                self._memory.add_tool_result(block.id, result)

            # --- Decide whether to stop ---
            if response.stop_reason == "end_turn":
                final_text = ""
                for block in response.content:
                    if block.type == "text":
                        final_text += block.text
                logger.info("Agent finished with end_turn")
                return final_text or "(No response text)"

            if response.stop_reason == "stop_sequence":
                logger.info("Agent stopped via stop_sequence")
                return "[Agent stopped — stop sequence encountered]"

            # Otherwise continue the loop (tool_use was the stop reason)
