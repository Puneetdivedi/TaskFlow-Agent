"""Shared fixtures for all test modules."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.memory.task_store import TaskStore


# ---------------------------------------------------------------------------
# Mock doubles reused across test modules
# ---------------------------------------------------------------------------
class MockLLMClient:
    """Minimal LLMClient stub — returns a configurable response."""

    def __init__(self) -> None:
        self._response = _make_end_turn_response("")

    async def send_messages(
        self,
        messages: list[dict] | None = None,
        system: str | None = None,
        tools: list[dict] | None = None,
    ) -> Any:
        return self._response


class MockMemory:
    """Minimal IMemory stub that tracks messages in-memory."""

    def __init__(self) -> None:
        self.messages: list[dict] = []
        self.history: list[tuple[str, str | list[dict]]] = []

    def add_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str | list[dict]) -> None:
        self.messages.append({"role": "assistant", "content": content})
        self.history.append(("assistant", content))

    def add_tool_result(self, tool_use_id: str, content: str) -> None:
        self.messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": content,
                    }
                ],
            }
        )

    def prune(self) -> None:
        pass

    def restore(self, messages: list[dict]) -> None:
        self.messages = list(messages)
        self.history.clear()

    def clear(self) -> None:
        self.messages.clear()
        self.history.clear()


class MockToolRegistry:
    """Minimal IToolRegistry stub — tracks invocation count."""

    def __init__(self) -> None:
        self._call_count = 0

    @property
    def tool_names(self) -> list[str]:
        return ["mock_tool"]

    def anthropic_tool_defs(self) -> list[dict]:
        return [
            {
                "name": "mock_tool",
                "description": "A mock tool",
                "input_schema": {"type": "object", "properties": {}},
            }
        ]

    async def dispatch(self, name: str, arguments: dict) -> str:
        self._call_count += 1
        return "mock result"


def _make_end_turn_response(text: str) -> Any:
    """Return an object duck-typing an Anthropic ``Message`` with end_turn."""
    content = [] if not text else [SimpleNamespace(type="text", text=text)]
    return SimpleNamespace(content=content, stop_reason="end_turn")


def _make_tool_use_response(tool_name: str, tool_input: dict, tool_id: str = "toolu_mock") -> Any:
    """Return an object duck-typing an Anthropic ``Message`` with a tool_use block."""
    block = SimpleNamespace(type="tool_use", id=tool_id, name=tool_name, input=tool_input)
    return SimpleNamespace(content=[block], stop_reason="tool_use")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_llm_client() -> MockLLMClient:
    return MockLLMClient()


@pytest.fixture
def mock_memory() -> MockMemory:
    return MockMemory()


@pytest.fixture
def mock_tool_registry() -> MockToolRegistry:
    return MockToolRegistry()


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """A temporary directory that is cleaned up after the test."""
    return tmp_path


@pytest.fixture
def sample_messages() -> list[dict]:
    """Return a typical short conversation for test reuse."""
    return [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "List files"},
        {"role": "assistant", "content": "Here are the files: ..."},
    ]


@pytest.fixture
def task_store(tmp_path: Path) -> TaskStore:
    """A TaskStore backed by a temporary JSON file."""
    return TaskStore(tasks_file=tmp_path / "tasks.json")
