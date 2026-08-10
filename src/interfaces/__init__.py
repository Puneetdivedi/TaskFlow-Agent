"""Abstract interfaces / protocols for dependency injection.

Every major component in the agent has a corresponding ``Protocol``
here so that implementations can be swapped (test doubles, alternative
LLM providers, custom memory backends) without modifying consumers.
"""

from __future__ import annotations

from src.interfaces.approval import ToolApprover
from src.interfaces.fact_store import Fact, IFactStore, format_facts
from src.interfaces.file_index import IFileIndex
from src.interfaces.llm_client import LLMClient, TextDeltaSink, ToolCallSink
from src.interfaces.memory import IMemory
from src.interfaces.session_store import ISessionStore, SessionInfo
from src.interfaces.task_store import ITaskStore, Task
from src.interfaces.tool_registry import IToolRegistry
from src.interfaces.usage import Usage

__all__ = [
    "Fact",
    "IFactStore",
    "IFileIndex",
    "IMemory",
    "ISessionStore",
    "ITaskStore",
    "IToolRegistry",
    "LLMClient",
    "SessionInfo",
    "Task",
    "TextDeltaSink",
    "ToolApprover",
    "ToolCallSink",
    "Usage",
    "format_facts",
]
