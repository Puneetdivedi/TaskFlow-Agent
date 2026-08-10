"""Tests for the remember/recall long-term memory tools."""

from __future__ import annotations

import pytest

from src.memory.fact_store import FactStore
from src.tools.base import ToolError
from src.tools.memory_tools import RecallTool, RememberTool


@pytest.fixture
def remember(fact_store: FactStore) -> RememberTool:
    return RememberTool(fact_store)


@pytest.fixture
def recall(fact_store: FactStore) -> RecallTool:
    return RecallTool(fact_store)


class TestRememberToolContract:
    def test_name_is_remember(self, remember: RememberTool) -> None:
        assert remember.name == "remember"

    def test_schema_requires_content(self, remember: RememberTool) -> None:
        schema = remember.input_schema
        assert schema["required"] == ["content"]
        assert "content" in schema["properties"]
        assert "topic" in schema["properties"]


class TestRememberTool:
    async def test_stores_and_reports_id(
        self, remember: RememberTool, fact_store: FactStore
    ) -> None:
        result = await remember.run(content="User prefers spaces")
        assert result == "Remembered fact f1: User prefers spaces"
        assert fact_store.get("f1").content == "User prefers spaces"

    async def test_empty_content_raises(self, remember: RememberTool) -> None:
        with pytest.raises(ToolError, match="must not be empty"):
            await remember.run(content="   ")

    async def test_stores_topic(self, remember: RememberTool, fact_store: FactStore) -> None:
        await remember.run(content="Uses PyCharm", topic="editor")
        assert fact_store.get("f1").topic == "editor"


class TestRecallToolContract:
    def test_name_is_recall(self, recall: RecallTool) -> None:
        assert recall.name == "recall"

    def test_schema_bounds_and_defaults(self, recall: RecallTool) -> None:
        schema = recall.input_schema
        assert "query" in schema["properties"]
        assert schema["properties"]["limit"]["minimum"] == 1
        assert schema["properties"]["limit"]["maximum"] == 100


class TestRecallTool:
    async def test_matches_query(self, recall: RecallTool, fact_store: FactStore) -> None:
        fact_store.add("User prefers spaces over tabs")
        fact_store.add("Deploy via pip")
        result = await recall.run(query="spaces")
        assert "1 memory:" in result
        assert "f1" in result
        assert "pip" not in result

    async def test_no_query_lists_recent(self, recall: RecallTool, fact_store: FactStore) -> None:
        fact_store.add("First")
        fact_store.add("Second")
        result = await recall.run()
        assert "2 memories:" in result
        assert "f2" in result
        assert "f1" in result

    async def test_empty_query_lists(self, recall: RecallTool, fact_store: FactStore) -> None:
        fact_store.add("First")
        result = await recall.run(query="")
        assert "f1" in result

    async def test_no_match_message(self, recall: RecallTool) -> None:
        assert (await recall.run(query="zzz")) == "No memories found."

    async def test_invalid_limit_raises(self, recall: RecallTool) -> None:
        with pytest.raises(ToolError, match="limit"):
            await recall.run(limit=0)
