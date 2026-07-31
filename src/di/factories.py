"""Production wiring — assembles dependencies for ``AgentOrchestrator``.

Call :func:`create_production_orchestrator` from the CLI entry point
to obtain a fully wired orchestrator ready for use.
"""

from __future__ import annotations

from config.settings import Settings
from src.agent.claude_client import ClaudeClient
from src.agent.orchestrator import AgentOrchestrator
from src.di.container import DIContainer
from src.interfaces import IMemory, IToolRegistry, LLMClient
from src.memory.conversation import ConversationMemory
from src.plugins import discover_tools
from src.tools.middleware import AuditMiddleware, LoggingMiddleware, ToolPipeline
from src.tools.registry import ToolRegistry


def _build_middleware() -> ToolPipeline:
    """Build the default middleware chain for production."""
    return ToolPipeline(middleware=[
        LoggingMiddleware(),
        AuditMiddleware(),
    ])


def register_defaults(container: DIContainer, settings: Settings) -> None:
    """Register the default (production) implementations with *container*."""

    container.register(LLMClient, lambda c: ClaudeClient(
        api_key=settings.anthropic_api_key,
        model=settings.anthropic_model,
    ))

    extra_tools = discover_tools()
    container.register(IToolRegistry, lambda c: ToolRegistry(
        work_dir=settings.work_dir,
        safety_level=settings.safety_level,
        extra_tools=extra_tools,
        pipeline=_build_middleware(),
    ))

    container.register(IMemory, lambda c: ConversationMemory(
        max_tokens=settings.max_history_tokens,
    ))


def create_production_orchestrator(
    settings: Settings | None = None,
) -> AgentOrchestrator:
    """Build and return a fully wired ``AgentOrchestrator``.

    Uses *settings* if provided, otherwise initialises from
    environment / config files.
    """
    if settings is None:
        from config.settings import init_settings
        settings = init_settings()

    container = DIContainer()
    register_defaults(container, settings)

    return AgentOrchestrator(
        llm_client=container.resolve(LLMClient),
        tools=container.resolve(IToolRegistry),
        memory=container.resolve(IMemory),
        max_tool_calls=settings.max_tool_calls_per_turn,
    )
