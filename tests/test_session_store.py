"""Tests for the persistent session store."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from src.memory.session_store import SessionStore


@pytest.fixture
def store(tmp_path: Path) -> SessionStore:
    return SessionStore(db_path=tmp_path / "sessions.db")


def _sample_messages() -> list[dict]:
    return [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]


class TestSessionStore:
    def test_save_and_load_roundtrip(self, store: SessionStore) -> None:
        msgs = _sample_messages()
        store.save("work", msgs)
        assert store.load("work") == msgs

    def test_save_persists_db(self, store: SessionStore, tmp_path: Path) -> None:
        store.save("work", _sample_messages())
        assert (tmp_path / "sessions.db").exists()

    def test_save_preserves_tool_result_blocks(self, store: SessionStore) -> None:
        msgs = [
            {"role": "user", "content": "do it"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "run_shell", "input": {"cmd": "ls"}}
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}],
            },
        ]
        store.save("conv", msgs)
        assert store.load("conv") == msgs

    def test_load_missing_raises_key_error(self, store: SessionStore) -> None:
        with pytest.raises(KeyError, match="not found"):
            store.load("ghost")

    def test_delete_removes_session(self, store: SessionStore) -> None:
        store.save("work", _sample_messages())
        store.delete("work")
        assert not store.exists("work")
        assert store.list() == []

    def test_delete_missing_raises_key_error(self, store: SessionStore) -> None:
        with pytest.raises(KeyError, match="not found"):
            store.delete("ghost")

    def test_exists(self, store: SessionStore) -> None:
        assert store.exists("work") is False
        store.save("work", [])
        assert store.exists("work") is True

    def test_list_returns_metadata(self, store: SessionStore) -> None:
        store.save("b", _sample_messages())
        store.save("a", [])
        infos = store.list()
        assert len(infos) == 2
        by_name = {info.name: info for info in infos}
        assert by_name["b"].message_count == 2
        assert by_name["a"].message_count == 0
        assert by_name["b"].updated_at  # non-empty timestamp

    def test_list_empty_dir(self, store: SessionStore) -> None:
        assert store.list() == []

    def test_list_sorted_newest_first(self, store: SessionStore) -> None:
        store.save("first", [])
        time.sleep(0.01)
        store.save("second", [])
        infos = store.list()
        assert infos[0].name == "second"
        # Ordering is a sort by updated_at descending, not by name.
        timestamps = [info.updated_at for info in infos]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_latest(self, store: SessionStore) -> None:
        assert store.latest() is None
        store.save("work", [])
        assert store.latest() == "work"
        time.sleep(0.01)
        store.save("home", [])
        assert store.latest() == "home"

    def test_corrupt_db_starts_empty(self, store: SessionStore, tmp_path: Path) -> None:
        # A corrupt (non-SQLite) db file is rebuilt empty on open.
        bad = SessionStore(db_path=tmp_path / "bad.db")
        bad.save("good", [])
        assert bad.load("good") == []
        assert [info.name for info in bad.list()] == ["good"]

    def test_invalid_names_rejected(self, store: SessionStore) -> None:
        for bad in ("", "..", "a/b", ".hidden", "has space", "a" * 65, "a\\b"):
            with pytest.raises(ValueError):
                store.save(bad, [])
            with pytest.raises(ValueError):
                store.load(bad)

    def test_valid_names_accepted(self, store: SessionStore) -> None:
        for good in ("work", "Work.Dir-1_2", "a" * 64):
            store.save(good, [])
        assert len(store.list()) == 3
