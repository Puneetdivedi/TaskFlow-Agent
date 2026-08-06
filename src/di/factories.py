"""Production wiring — assembles dependencies for ``AgentOrchestrator``.

Call :func:`create_production_app` from the CLI entry point to obtain a
fully wired orchestrator, session store, and task store.
:func:`create_production_orchestrator` remains available for callers that
only need the orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass

from config.settings import Settings
from src.agent.claude_client import ClaudeClient
from src.agent.orchestrator import AgentOrchestrator
from src.di.container import DIContainer
from src.interfaces import IMemory, ISessionStore, ITaskStore, IToolRegistry, LLMClient
from src.memory.conversation import ConversationMemory
from src.memory.migration import migrate_legacy_data
from src.memory.session_store import SessionStore
from src.memory.sqlite_store import DEFAULT_DB_PATH
from src.memory.task_store import TaskStore
from src.plugins import discover_tools
from src.tools.middleware import AuditMiddleware, LoggingMiddleware, ToolPipeline
from src.tools.registry import ToolRegistry


@dataclass(frozen=True)
class AppComponents:
    """The fully wired application: orchestrator plus its stores."""

    orchestrator: AgentOrchestrator
    session_store: ISessionStore
    task_store: ITaskStore


def _build_middleware() -> ToolPipeline:
    """Build the default middleware chain for production."""
    return ToolPipeline(
        middleware=[
            LoggingMiddleware(),
            AuditMiddleware(),
        ]
    )


def register_defaults(container: DIContainer, settings: Settings) -> None:
    """Register the default (production) implementations with *container*."""

    # One-time migration of legacy JSON data into the shared SQLite database.
    # Only the default database location migrates, so tests (which use
    # temporary db paths) never pull in real ~/.taskflow data.
    if settings.db_path == DEFAULT_DB_PATH:
        migrate_legacy_data(settings.db_path)

    container.register(
        LLMClient,
        lambda c: ClaudeClient(
            api_key=settings.anthropic_api_key,
            model=settings.anthropic_model,
        ),
    )

    extra_tools = discover_tools()
    container.register(
        IToolRegistry,
        lambda c: ToolRegistry(
            work_dir=settings.work_dir,
            safety_level=settings.safety_level,
            extra_tools=extra_tools,
            pipeline=_build_middleware(),
            task_store=c.resolve(ITaskStore),
            file_index_db=settings.db_path,
        ),
    )

    container.register(
        IMemory,
        lambda c: ConversationMemory(
            max_tokens=settings.max_history_tokens,
        ),
    )

    container.register(
        ISessionStore,
        lambda c: SessionStore(db_path=settings.db_path),
    )

    container.register(
        ITaskStore,
        lambda c: TaskStore(db_path=settings.db_path),
    )


def create_production_app(settings: Settings | None = None) -> AppComponents:
    """Build and return a fully wired orchestrator and session store.

    Uses *settings* if provided, otherwise initialises from
    environment / config files.
    """
    if settings is None:
        from config.settings import init_settings

        settings = init_settings()

    container = DIContainer()
    register_defaults(container, settings)

    orchestrator = AgentOrchestrator(
        llm_client=container.resolve(LLMClient),
        tools=container.resolve(IToolRegistry),
        memory=container.resolve(IMemory),
        max_tool_calls=settings.max_tool_calls_per_turn,
    )
    return AppComponents(
        orchestrator=orchestrator,
        session_store=container.resolve(ISessionStore),
        task_store=container.resolve(ITaskStore),
    )


def create_production_orchestrator(
    settings: Settings | None = None,
) -> AgentOrchestrator:
    """Build and return a fully wired ``AgentOrchestrator``.

    Uses *settings* if provided, otherwise initialises from
    environment / config files.
    """
    return create_production_app(settings).orchestrator
