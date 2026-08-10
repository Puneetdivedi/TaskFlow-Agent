"""Tests for the persistent fact (cross-session memory) store."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.interfaces.fact_store import Fact, format_facts
from src.memory.fact_store import FactStore


class TestFactStore:
    def test_add_returns_fact(self, fact_store: FactStore) -> None:
        fact = fact_store.add("User prefers spaces over tabs")
        assert fact.id == "f1"
        assert fact.content == "User prefers spaces over tabs"
        assert fact.topic == ""
        assert fact.created_at

    def test_add_strips_whitespace(self, fact_store: FactStore) -> None:
        fact = fact_store.add("  trimmed  ")
        assert fact.content == "trimmed"

    def test_add_with_topic(self, fact_store: FactStore) -> None:
        fact = fact_store.add("Uses PyCharm", topic="editor")
        assert fact.topic == "editor"

    def test_add_increments_ids(self, fact_store: FactStore) -> None:
        fact_store.add("First")
        second = fact_store.add("Second")
        assert second.id == "f2"

    def test_add_rejects_empty_content(self, fact_store: FactStore) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            fact_store.add("   ")

    def test_list_empty(self, fact_store: FactStore) -> None:
        assert fact_store.list() == []

    def test_list_newest_first(self, fact_store: FactStore) -> None:
        fact_store.add("First")
        fact_store.add("Second")
        fact_store.add("Third")
        assert [f.id for f in fact_store.list()] == ["f3", "f2", "f1"]

    def test_list_limit(self, fact_store: FactStore) -> None:
        for i in range(5):
            fact_store.add(f"Fact {i}")
        assert [f.id for f in fact_store.list(limit=2)] == ["f5", "f4"]

    def test_search_by_content_substring(self, fact_store: FactStore) -> None:
        fact_store.add("User prefers spaces over tabs")
        fact_store.add("Deploy with pip install")
        assert [f.id for f in fact_store.search("spaces")] == ["f1"]

    def test_search_is_case_insensitive(self, fact_store: FactStore) -> None:
        fact_store.add("User prefers SPACES over tabs")
        assert [f.id for f in fact_store.search("spaces")] == ["f1"]

    def test_search_matches_topic(self, fact_store: FactStore) -> None:
        fact_store.add("Uses PyCharm", topic="editor")
        assert [f.id for f in fact_store.search("editor")] == ["f1"]

    def test_search_empty_query_lists(self, fact_store: FactStore) -> None:
        fact_store.add("First")
        fact_store.add("Second")
        assert [f.id for f in fact_store.search("")] == ["f2", "f1"]

    def test_search_no_match_returns_empty(self, fact_store: FactStore) -> None:
        fact_store.add("First")
        assert fact_store.search("zzz") == []

    def test_search_limit(self, fact_store: FactStore) -> None:
        for i in range(3):
            fact_store.add(f"shared word {i}")
        assert len(fact_store.search("word", limit=2)) == 2

    def test_invalid_limit_rejected(self, fact_store: FactStore) -> None:
        with pytest.raises(ValueError, match="limit"):
            fact_store.list(limit=0)
        with pytest.raises(ValueError, match="limit"):
            fact_store.list(limit=True)
        with pytest.raises(ValueError, match="limit"):
            fact_store.search("x", limit=-1)

    def test_get_returns_fact(self, fact_store: FactStore) -> None:
        fact_store.add("First")
        assert fact_store.get("f1").content == "First"

    def test_get_missing_raises_key_error(self, fact_store: FactStore) -> None:
        with pytest.raises(KeyError, match="not found"):
            fact_store.get("f99")

    def test_delete_removes_fact(self, fact_store: FactStore) -> None:
        fact_store.add("First")
        fact_store.delete("f1")
        assert fact_store.list() == []

    def test_delete_missing_raises_key_error(self, fact_store: FactStore) -> None:
        with pytest.raises(KeyError, match="not found"):
            fact_store.delete("f99")

    def test_reload_persists_roundtrip(self, fact_store: FactStore, tmp_path: Path) -> None:
        fact_store.add("User prefers spaces", topic="prefs")
        reloaded = FactStore(db_path=tmp_path / "facts.db")
        facts = reloaded.list()
        assert len(facts) == 1
        assert facts[0].content == "User prefers spaces"
        assert facts[0].topic == "prefs"

    def test_corrupt_db_starts_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "facts.db"
        path.write_text("this is not a sqlite database", encoding="utf-8")
        store = FactStore(db_path=path)
        assert store.list() == []


class TestFormatFacts:
    def test_empty(self) -> None:
        assert format_facts([]) == "No memories found."

    def test_single_fact(self) -> None:
        fact = Fact(id="f1", content="hello", topic="", created_at="")
        assert format_facts([fact]) == "1 memory:\n  [f1] hello"

    def test_multiple_facts_with_topics(self) -> None:
        facts = [
            Fact(id="f1", content="hello", topic="", created_at=""),
            Fact(id="f2", content="world", topic="greeting", created_at=""),
        ]
        text = format_facts(facts)
        assert "2 memories:" in text
        assert "[f1] hello" in text
        assert "[f2] greeting: world" in text
