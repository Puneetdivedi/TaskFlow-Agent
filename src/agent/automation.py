"""Autonomous task execution — a fresh, bounded run of a task's plan.

When a task carries a ``plan`` and is due (or is run via ``/task run`` or the
``automation_run`` tool), an :class:`AutomationRunner` executes that plan to
completion in its own isolated, bounded tool loop — the same pattern
:class:`SubAgentRunner` uses for sub-agents, but for user-authored task plans.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from src.agent.orchestrator import AgentOrchestrator
from src.interfaces import IFactStore, IToolRegistry, LLMClient
from src.memory.conversation import ConversationMemory

logger = logging.getLogger(__name__)

#: Default per-run tool-call budget for an autonomous run (the runaway-loop
#: backstop — mirror of the orchestrator's default).
_DEFAULT_MAX_TOOL_CALLS = 25

_AUTOMATION_SYSTEM_PROMPT = """\
You are **TaskFlow Agent**, running a scheduled task autonomously on behalf of \
the user.

The user wrote the plan below and expects it executed to completion without \
further input. Work through the steps in order, call tools as needed, and when \
you are done, return a short report of what you did and the outcome.

## Constraints
- Only use tools you are allowed. Destructive or outward-facing actions \
(delete, move, shell, email) are automatically denied unless this task has \
opted into full autonomy — if a step is denied, say what you could not do and \
why.
- Never ask the user for clarification or confirmation; this run is unattended.
- If a step fails, try an alternative approach, then explain the limitation.
- Be concise but informative.
"""


class AutomationRunner:
    """Runs a task's ``plan`` to completion in an isolated, bounded loop.

    Each run builds a **fresh** :class:`ConversationMemory` and a **fresh**
    :class:`AgentOrchestrator` over the shared client and registry, so an
    autonomous execution never pollutes — or is polluted by — the user's live
    conversation. The plan is seeded as the only user message and executed
    under the autonomous approval policy (destructive ops denied unless the
    task opts into full autonomy).
    """

    def __init__(
        self,
        llm_client: LLMClient,
        registry: IToolRegistry,
        fact_store: IFactStore | None = None,
        *,
        max_tool_calls: int = _DEFAULT_MAX_TOOL_CALLS,
        allow_destructive: bool = False,
        system_prompt: str | None = None,
    ) -> None:
        self._client = llm_client
        self._registry = registry
        self._fact_store = fact_store
        self._max_tool_calls = max_tool_calls
        self._allow_destructive = allow_destructive
        self._system_prompt = system_prompt or _AUTOMATION_SYSTEM_PROMPT

    # ------------------------------------------------------------------
    @property
    def registry(self) -> IToolRegistry:
        return self._registry

    # ------------------------------------------------------------------
    async def run_plan(self, plan: str, title: str = "") -> str:
        """Execute *plan* and return a formatted report string.

        Never raises: LLM/tool failures are turned into report text, so the
        caller — the CLI scheduler or the ``automation_run`` tool — always gets
        a string it can show or store. On ``KeyboardInterrupt`` the run is
        aborted and reported as a failure (the orchestrator rolls its own
        memory back before re-raising).
        """
        start_time = datetime.now()
        memory = ConversationMemory()
        orchestrator = AgentOrchestrator(
            llm_client=self._client,
            tools=self._registry,
            memory=memory,
            system_prompt=self._system_prompt,
            max_tool_calls=self._max_tool_calls,
            fact_store=self._fact_store,
        )
        # Imported lazily: ``src.ui``'s package ``__init__`` pulls in ``cli``,
        # which imports this module — a module-level import here would cycle.
        from src.ui.approvals import autonomous_approver

        user_input = plan if not title else f"# {title}\n\n{plan}"
        try:
            result = await orchestrator.run(
                user_input,
                tool_approver=autonomous_approver(allow_destructive=self._allow_destructive),
            )
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            return self._format_report(
                title=title or "Automation Run",
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                plan=plan,
                result=result,
                success=True,
            )
        except KeyboardInterrupt:
            logger.warning("Automation run aborted by the user")
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            return self._format_report(
                title=title or "Automation Run",
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                plan=plan,
                result="(Automation aborted by the user.)",
                success=False,
            )
        except Exception as exc:  # noqa: BLE001 — always degrade to a report string
            logger.warning("Automation run failed: %s", exc)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            return self._format_report(
                title=title or "Automation Run",
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                plan=plan,
                result=f"(Automation failed: {exc})",
                success=False,
            )

    def _format_report(
        self,
        *,
        title: str,
        start_time: datetime,
        end_time: datetime,
        duration_seconds: float,
        plan: str,
        result: str,
        success: bool,
    ) -> str:
        """Format a structured automation run report."""
        status = "✅ SUCCESS" if success else "❌ FAILED"
        lines = [
            f"# Automation Report: {title}",
            f"",
            f"**Status:** {status}",
            f"**Started:** {start_time.isoformat(timespec='seconds')}",
            f"**Completed:** {end_time.isoformat(timespec='seconds')}",
            f"**Duration:** {duration_seconds:.1f}s",
            f"",
            f"## Plan",
            f"```",
            plan.strip() if plan else "(no plan provided)",
            f"```",
            f"",
            f"## Result",
            result.strip() if result else "(no output)",
        ]
        return "\n".join(lines)