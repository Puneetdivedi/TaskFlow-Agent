"""Main agent loop — decides *what* to do, calls tools, and returns answers."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from src.interfaces import (
    IMemory,
    IToolRegistry,
    LLMClient,
    TextDeltaSink,
    ToolCallSink,
)
from src.interfaces.usage import Usage
from src.memory.summary import (
    ConversationSummarizer,
    format_turns,
    plan_consolidation,
)
from src.memory.tokens import estimate_tokens

logger = logging.getLogger(__name__)

# Default system prompt that shapes the agent's personality and constraints.
DEFAULT_SYSTEM_PROMPT = """\
You are **TaskFlow Agent**, an autonomous AI assistant that helps users automate \
their day-to-day tasks.

## Your capabilities
You have access to tools that let you:
- Read, write, list, search, move, and delete files
- Execute shell commands (with safety restrictions)
- Search the web and fetch pages (web_search / web_fetch)
- Read and write structured YAML files (yaml_read / yaml_write)

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
        max_parallel_tool_calls: int = 5,
        summarizer: ConversationSummarizer | None = None,
        summary_threshold_tokens: int = 0,
        cost_budget_usd: float = 0.0,
    ) -> None:
        self._client = llm_client
        self._tools = tools
        self._memory = memory
        self._system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self._max_tool_calls = max_tool_calls
        self._max_parallel_tool_calls = max_parallel_tool_calls
        self._summarizer = summarizer
        self._summary_threshold_tokens = summary_threshold_tokens
        self._cost_budget_usd = cost_budget_usd

    # ------------------------------------------------------------------
    @property
    def tools(self) -> IToolRegistry:
        return self._tools

    @property
    def memory(self) -> IMemory:
        return self._memory

    @property
    def usage(self) -> Usage:
        """Cumulative token usage across the process's LLM calls.

        Reads the shared client's accumulator when it exposes one (the
        production ``ClaudeClient`` does); test doubles without it degrade to
        an empty ``Usage``.
        """
        usage = getattr(self._client, "usage", None)
        return usage if isinstance(usage, Usage) else Usage()

    def estimated_cost(self) -> float:
        """Estimated USD cost of the accumulated usage (0.0 when unknown)."""
        cost = getattr(self._client, "estimated_cost", None)
        return cost() if callable(cost) else 0.0

    # ------------------------------------------------------------------
    async def _dispatch_safe(self, block: Any) -> tuple[str, str]:
        """Run one tool and return ``(tool_use_id, result)``.

        Any exception — not just ``ToolError`` — becomes an ``Error: ...``
        result so a single failing tool can't cancel the concurrent batch or
        reorder the recorded results.
        """
        try:
            result = await self._tools.dispatch(block.name, block.input)
        except Exception as exc:  # noqa: BLE001 — isolate tool failures from the gather
            result = f"Error: {exc}"
        logger.debug("Tool result for %s: %.500s", block.name, result)
        return block.id, result

    # ------------------------------------------------------------------
    def _system_blocks(self) -> str | list[dict[str, str]]:
        """Return the ``system`` argument for LLM calls.

        A plain string (the base prompt) when there is no rolling summary;
        otherwise a block list that injects the summary ahead of the base
        prompt, e.g.::

            [{"type": "text", "text": "[Summary of earlier conversation]\\n..."},
             {"type": "text", "text": "<base prompt>"}]

        The Anthropic API accepts both forms.
        """
        if not self._memory.summary:
            return self._system_prompt
        return [
            {
                "type": "text",
                "text": f"[Summary of earlier conversation]\n{self._memory.summary}",
            },
            {"type": "text", "text": self._system_prompt},
        ]

    async def _maybe_consolidate(self) -> None:
        """Condense old turns into a rolling summary once the conversation
        grows past ``summary_threshold_tokens``.

        Summarizes the older prefix (chosen so no exchange is split), drops it
        from memory, and stores the result on ``IMemory.summary`` so it is
        injected as a system block on subsequent calls. A failed summary is
        logged and ignored — memory is left untouched and the turn proceeds
        with the full history.
        """
        if self._summarizer is None or self._summary_threshold_tokens <= 0:
            return
        messages = self._memory.messages
        if estimate_tokens(str(messages)) <= self._summary_threshold_tokens:
            return

        keep_recent_tokens = max(2000, self._summary_threshold_tokens // 2)
        to_summarize, keep = plan_consolidation(messages, keep_recent_tokens=keep_recent_tokens)
        if not to_summarize:
            return

        try:
            new_summary = await self._summarizer.summarize(
                format_turns(to_summarize), existing=self._memory.summary
            )
        except Exception as exc:  # noqa: BLE001 — never let a failed summary break the turn
            logger.warning("Semantic-memory consolidation skipped: %s", exc)
            return

        self._memory.restore(keep)
        self._memory.set_summary(new_summary)
        logger.info(
            "Consolidated %d old message(s) into a summary (%d kept)",
            len(to_summarize),
            len(keep),
        )

    # ------------------------------------------------------------------
    async def run(
        self,
        user_input: str,
        *,
        on_text_delta: TextDeltaSink | None = None,
        on_tool_call: ToolCallSink | None = None,
    ) -> str:
        """Process a single user request through the full tool-calling loop.

        When *on_text_delta* is provided the assistant's text is streamed
        through it as the model generates it; otherwise the whole response
        arrives at once. *on_tool_call* (if given) is awaited with
        ``(name, args)`` immediately before each tool runs. On
        ``KeyboardInterrupt`` the turn's conversation changes are rolled
        back before the interrupt propagates, so an aborted turn leaves
        memory clean.

        Returns the final assistant response string.
        """
        # Compact long conversations into a rolling summary before recording
        # this turn. The snapshot below is taken *after* consolidation so a
        # KeyboardInterrupt rollback keeps the (legitimate) memory compaction.
        await self._maybe_consolidate()

        # Snapshot before recording the user message so an aborted turn rolls
        # back completely (the user input included), leaving the conversation
        # exactly as it was before the turn started.
        before = list(self._memory.messages)

        self._memory.add_user(user_input)
        logger.info("Processing user input: %.200s", user_input)

        tool_call_count = 0

        try:
            while True:
                try:
                    if on_text_delta is not None:
                        response = await self._client.stream_messages(
                            messages=self._memory.messages,
                            system=self._system_blocks(),
                            tools=self._tools.anthropic_tool_defs(),
                            on_text_delta=on_text_delta,
                        )
                    else:
                        response = await self._client.send_messages(
                            messages=self._memory.messages,
                            system=self._system_blocks(),
                            tools=self._tools.anthropic_tool_defs(),
                        )
                except Exception as exc:  # broad catch — interface impls may raise different errors
                    error_text = f"API error: {exc}"
                    logger.error("API call failed: %s", exc)
                    self._memory.add_assistant(error_text)
                    return error_text

                # --- Respect the optional cost budget: stop before running tools ---
                if self._cost_budget_usd > 0 and self.estimated_cost() >= self._cost_budget_usd:
                    budget_msg = (
                        f"Budget exhausted — estimated spend "
                        f"${self.estimated_cost():.4f} has reached the "
                        f"${self._cost_budget_usd:.2f} cap, so I'm stopping here."
                    )
                    logger.warning(
                        "Cost budget exhausted: $%.4f >= $%.2f",
                        self.estimated_cost(),
                        self._cost_budget_usd,
                    )
                    self._memory.add_assistant(budget_msg)
                    return budget_msg

                # --- Build a single assistant response from all content blocks ---
                assistant_content: list[dict[str, Any]] = []
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
                # Blocks beyond the max_tool_calls budget are rejected up front
                # with the abort message (the halt still fires after exactly
                # ``max_tool_calls`` tools have run). The rest dispatch in
                # chunks of ``max_parallel_tool_calls``; their results are
                # recorded together, in block order, in a single user message.
                remaining = self._max_tool_calls - tool_call_count
                run, aborted = tool_blocks[:remaining], tool_blocks[remaining:]

                if run:
                    results: list[tuple[str, str]] = []
                    for start in range(0, len(run), self._max_parallel_tool_calls):
                        chunk = run[start : start + self._max_parallel_tool_calls]
                        for block in chunk:
                            tool_call_count += 1
                            logger.info(
                                "Dispatching tool: %s with args: %s", block.name, block.input
                            )
                            if on_tool_call is not None:
                                await on_tool_call(block.name, block.input)
                        results.extend(
                            await asyncio.gather(*(self._dispatch_safe(block) for block in chunk))
                        )
                    self._memory.add_tool_results(results)

                if aborted:
                    error_msg = (
                        f"Exceeded max tool calls ({self._max_tool_calls}). "
                        "Aborting to prevent runaway execution."
                    )
                    logger.warning("Max tool calls exceeded: %d", self._max_tool_calls)
                    self._memory.add_tool_results([(block.id, error_msg) for block in aborted])
                    return error_msg

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
        except KeyboardInterrupt:
            # Roll the current turn back so an aborted response (and any tool
            # results it produced) is not kept as conversation history.
            self._memory.restore(before)
            raise
