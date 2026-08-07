"""Sub-agents — isolated, bounded tool-calling loops the main agent can delegate to.

A sub-agent is a named role with its own system prompt, a restricted tool set,
and a fresh context. When the main agent calls the ``subagent`` tool, a
:class:`SubAgentRunner` drives that role's own short tool loop to completion and
returns a final report string. The sub-agent never sees the main conversation,
and (by default) it is never offered the ``subagent`` tool itself, so it cannot
spawn unbounded descendants.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from src.interfaces import IToolRegistry, LLMClient

logger = logging.getLogger(__name__)

#: Name of the tool that exposes sub-agents to the main agent. Deliberately
#: excluded from a sub-agent's "all tools" default to prevent unbounded nesting.
SUBAGENT_TOOL_NAME = "subagent"

#: Default per-run step budget for a sub-agent loop (a sub-agent is a bounded,
#: short-lived task — the budget is the runaway-loop backstop).
_DEFAULT_MAX_STEPS = 12

_RESEARCHER_PROMPT = """\
You are **Researcher**, a sub-agent focused on gathering and verifying
information. Be precise and cite your sources. Do not fabricate facts — if you
cannot confirm something, say so plainly. When you have enough to answer the
task, stop and return a concise report with the key findings and their sources.
"""

_CODER_PROMPT = """\
You are **Coder**, a sub-agent focused on implementing changes to the user's
codebase. Read the relevant files first, then make minimal, correct changes.
Prefer small, focused edits. If you can, run a quick check to verify your work.
Return a short summary of what you changed, plus any limitations or open
questions.
"""

_REVIEWER_PROMPT = """\
You are **Reviewer**, a sub-agent focused on reviewing code or documents for
correctness, security, and clarity. Read the relevant files, look for real
issues (bugs, unsafe patterns, style violations), and return findings ranked
by severity. Be specific — cite file and line where possible. Do not restate
the code you reviewed.
"""


@dataclass(frozen=True)
class SubAgent:
    """A named sub-agent role.

    ``tools`` is the set of tool names the role may see and run. An empty tuple
    means "all registered tools except the ``subagent`` tool itself".
    ``max_tool_calls`` bounds one delegation (the runner's own budget is the
    upper bound).
    """

    name: str
    system_prompt: str
    tools: tuple[str, ...] = ()
    max_tool_calls: int = 15


def default_subagents() -> dict[str, SubAgent]:
    """Return the built-in sub-agent roles: researcher, coder, reviewer."""
    return {
        "researcher": SubAgent(
            name="researcher",
            system_prompt=_RESEARCHER_PROMPT,
            tools=("web_search", "web_fetch", "read_file", "list_files", "search_files"),
        ),
        "coder": SubAgent(
            name="coder",
            system_prompt=_CODER_PROMPT,
            tools=(
                "read_file",
                "write_file",
                "list_files",
                "search_files",
                "move_file",
                "delete_file",
                "run_shell",
                "yaml_read",
                "yaml_write",
            ),
        ),
        "reviewer": SubAgent(
            name="reviewer",
            system_prompt=_REVIEWER_PROMPT,
            tools=("read_file", "list_files", "search_files", "run_shell"),
        ),
    }


class SubAgentRunner:
    """Drives one named sub-agent to completion in an isolated tool loop.

    Each delegation builds a fresh message list seeded with the task, offers the
    role only its allowed tools, and dispatches tool calls through the shared
    tool registry (so middleware, plugins, and MCP tools behave as usual).
    """

    def __init__(
        self,
        llm_client: LLMClient,
        registry: IToolRegistry,
        subagents: dict[str, SubAgent],
        *,
        max_steps: int = _DEFAULT_MAX_STEPS,
    ) -> None:
        self._client = llm_client
        self._registry = registry
        self._subagents = subagents
        self._max_steps = max_steps

    # ------------------------------------------------------------------
    @property
    def agent_names(self) -> list[str]:
        return list(self._subagents.keys())

    def _allowed_for(self, spec: SubAgent) -> set[str]:
        if spec.tools:
            return set(spec.tools)
        return set(self._registry.tool_names) - {SUBAGENT_TOOL_NAME}

    def _defs_for(self, spec: SubAgent) -> list[dict[str, Any]]:
        """Anthropic tool defs filtered to the role's allowed tools."""
        allowed = self._allowed_for(spec)
        return [d for d in self._registry.anthropic_tool_defs() if d.get("name") in allowed]

    # ------------------------------------------------------------------
    async def run(self, name: str, task: str) -> str:
        """Run the named sub-agent on *task* and return its final report.

        Raises ``KeyError`` for an unknown sub-agent name. LLM/tool failures are
        turned into report text (never a raised exception), so the caller — the
        main agent — always gets a string it can reason about.
        """
        spec = self._subagents[name]
        messages: list[dict[str, Any]] = [{"role": "user", "content": task}]
        tool_defs = self._defs_for(spec)
        allowed = self._allowed_for(spec)
        budget = max(1, min(self._max_steps, spec.max_tool_calls))

        for _ in range(budget):
            try:
                response = await self._client.send_messages(
                    messages=messages,
                    system=spec.system_prompt,
                    tools=tool_defs,
                )
            except Exception as exc:
                logger.warning("Sub-agent %r failed mid-loop: %s", name, exc)
                return f"(Sub-agent '{name}' failed with an API error: {exc})"

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

            # Preserve the assistant entry even when it only carried tool calls,
            # so the user → assistant → tool_result alternation stays valid.
            if assistant_content:
                messages.append({"role": "assistant", "content": assistant_content})

            if response.stop_reason == "end_turn":
                final_text = "".join(
                    block.text for block in response.content if block.type == "text"
                )
                logger.info("Sub-agent %r finished (%d steps)", name, budget)
                return final_text or "(no text)"

            if response.stop_reason == "stop_sequence":
                return "[Sub-agent stopped — stop sequence encountered]"

            for block in tool_blocks:
                if block.name not in allowed:
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": block.id,
                                    "content": (
                                        f"Error: tool '{block.name}' is not available "
                                        f"to this sub-agent."
                                    ),
                                }
                            ],
                        }
                    )
                    continue
                try:
                    result = await self._registry.dispatch(block.name, block.input)
                except Exception as exc:
                    result = f"Error: {exc}"
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result,
                            }
                        ],
                    }
                )

        logger.warning("Sub-agent %r hit its step budget (%d)", name, budget)
        return (
            f"(Sub-agent '{name}' reached its step budget ({budget}) and was "
            "truncated before finishing.)"
        )
