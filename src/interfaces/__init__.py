"""Abstract interfaces / protocols for dependency injection.

Every major component in the agent has a corresponding ``Protocol``
here so that implementations can be swapped (test doubles, alternative
LLM providers, custom memory backends) without modifying consumers.
"""

from __future__ import annotations

from src.interfaces.file_index import IFileIndex
from src.interfaces.llm_client import LLMClient
from src.interfaces.memory import IMemory
from src.interfaces.session_store import ISessionStore, SessionInfo
from src.interfaces.task_store import ITaskStore, Task
from src.interfaces.tool_registry import IToolRegistry

__all__ = [
    "IFileIndex",
    "IMemory",
    "ISessionStore",
    "ITaskStore",
    "IToolRegistry",
    "LLMClient",
    "SessionInfo",
    "Task",
]
