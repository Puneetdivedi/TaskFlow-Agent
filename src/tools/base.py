"""Base class and type definitions for all agent tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ToolError(Exception):
    """Raised when a tool execution fails."""


class Tool(ABC):
    """A single tool the agent can invoke."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short unique name — matches what the LLM calls via tool_use.name."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Natural-language description shown to the LLM so it knows when to use this tool."""

    @property
    @abstractmethod
    def input_schema(self) -> dict:
        """JSON Schema for the tool's input parameters."""

    @abstractmethod
    async def run(self, **kwargs: Any) -> str:
        """Execute the tool and return a string result for the LLM.

        Every tool *must* return a plain string —
        the LLM never sees raw objects, only this text.
        """
