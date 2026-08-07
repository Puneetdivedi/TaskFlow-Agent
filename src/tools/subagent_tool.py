"""Tool that lets the main agent delegate a task to a named sub-agent."""

from __future__ import annotations

import logging
from typing import Any

from src.agent.subagent import SubAgentRunner
from src.tools.base import Tool

logger = logging.getLogger(__name__)


class SubAgentTool(Tool):
    """Delegate a self-contained subtask to a specialized sub-agent.

    The runner runs the requested role in its own isolated, bounded tool loop
    and returns a final report string. Failures are returned as error text —
    never raised — so the main agent can keep reasoning about the conversation.
    """

    def __init__(self, runner: SubAgentRunner) -> None:
        self._runner = runner
        self._names = tuple(runner.agent_names)

    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return "subagent"

    @property
    def description(self) -> str:
        names = ", ".join(self._names)
        return (
            "Delegate a self-contained, focused subtask to a specialized "
            "sub-agent. Each sub-agent runs its own isolated context with a "
            "restricted tool set and returns a final report. Use this for a "
            "distinct phase of a larger request (research, coding, review) that "
            f"is cleaner to run in isolation. Available sub-agents: {names}."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "enum": list(self._names),
                    "description": "Which sub-agent to delegate to.",
                },
                "task": {
                    "type": "string",
                    "description": (
                        "A self-contained task for the sub-agent. Be specific: "
                        "what to do, what to read or write, and what to return."
                    ),
                },
            },
            "required": ["name", "task"],
        }

    # ------------------------------------------------------------------
    async def run(self, *, name: str, task: str) -> str:  # type: ignore[override]
        """Delegate and return the sub-agent's final report (or an error)."""
        try:
            return await self._runner.run(name, task)
        except KeyError:
            return (
                f"Error: unknown sub-agent {name!r}. Available sub-agents: {', '.join(self._names)}"
            )
        except Exception as exc:
            logger.warning("Sub-agent %r failed unexpectedly: %s", name, exc)
            return f"Error: sub-agent '{name}' failed: {exc}"
