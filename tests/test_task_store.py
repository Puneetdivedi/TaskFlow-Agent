"""Tests for the persistent task store."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from src.memory.task_store import TaskStore


class TestTaskStore:
    def test_create_returns_task(self, task_store: TaskStore) -> None:
        task = task_store.create("Buy milk")
        assert task.id == "t1"
        assert task.title == "Buy milk"
        assert task.status == "todo"
        assert task.priority == "medium"
        assert task.created_at
        assert task.updated_at

    def test_create_persists_json_file(self, task_store: TaskStore, tmp_path: Path) -> None:
        task_store.create("Buy milk")
        assert (tmp_path / "tasks.json").exists()

    def test_create_increments_ids(self, task_store: TaskStore) -> None:
        task_store.create("First")
        second = task_store.create("Second")
        assert second.id == "t2"

    def test_create_rejects_empty_title(self, task_store: TaskStore) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            task_store.create("   ")

    def test_create_rejects_bad_priority(self, task_store: TaskStore) -> None:
        with pytest.raises(ValueError, match="Invalid priority"):
            task_store.create("Task", priority="urgent")

    def test_get_returns_task(self, task_store: TaskStore) -> None:
        task_store.create("Buy milk")
        assert task_store.get("t1").title == "Buy milk"

    def test_get_missing_raises_key_error(self, task_store: TaskStore) -> None:
        with pytest.raises(KeyError, match="not found"):
            task_store.get("t99")

    def test_list_empty(self, task_store: TaskStore) -> None:
        assert task_store.list() == []

    def test_list_newest_first(self, task_store: TaskStore) -> None:
        task_store.create("First")
        time.sleep(0.01)
        task_store.create("Second")
        assert [t.id for t in task_store.list()] == ["t2", "t1"]

    def test_list_filters_by_status(self, task_store: TaskStore) -> None:
        task_store.create("First")
        task_store.create("Second")
        task_store.update("t1", status="done")
        done = task_store.list("done")
        todo = task_store.list("todo")
        assert [t.id for t in done] == ["t1"]
        assert [t.id for t in todo] == ["t2"]

    def test_list_invalid_status_raises(self, task_store: TaskStore) -> None:
        with pytest.raises(ValueError, match="Invalid status"):
            task_store.list("bogus")

    def test_update_fields(self, task_store: TaskStore) -> None:
        task_store.create("Buy milk", description="2L")
        updated = task_store.update(
            "t1",
            title="Buy oat milk",
            status="in_progress",
            priority="high",
        )
        assert updated.title == "Buy oat milk"
        assert updated.description == "2L"  # untouched field preserved
        assert updated.status == "in_progress"
        assert updated.priority == "high"

    def test_update_no_changes_returns_unchanged(self, task_store: TaskStore) -> None:
        created = task_store.create("Buy milk")
        updated = task_store.update("t1")
        assert updated.title == created.title
        assert updated.updated_at == created.updated_at

    def test_update_missing_raises_key_error(self, task_store: TaskStore) -> None:
        with pytest.raises(KeyError, match="not found"):
            task_store.update("t99", status="done")

    def test_update_invalid_status_raises(self, task_store: TaskStore) -> None:
        task_store.create("Buy milk")
        with pytest.raises(ValueError, match="Invalid status"):
            task_store.update("t1", status="bogus")

    def test_update_invalid_priority_raises(self, task_store: TaskStore) -> None:
        task_store.create("Buy milk")
        with pytest.raises(ValueError, match="Invalid priority"):
            task_store.update("t1", priority="bogus")

    def test_update_empty_title_raises(self, task_store: TaskStore) -> None:
        task_store.create("Buy milk")
        with pytest.raises(ValueError, match="must not be empty"):
            task_store.update("t1", title="  ")

    def test_delete_removes_task(self, task_store: TaskStore) -> None:
        task_store.create("Buy milk")
        task_store.delete("t1")
        assert task_store.list() == []

    def test_delete_missing_raises_key_error(self, task_store: TaskStore) -> None:
        with pytest.raises(KeyError, match="not found"):
            task_store.delete("t99")

    def test_reload_persists_roundtrip(self, task_store: TaskStore, tmp_path: Path) -> None:
        task_store.create("Buy milk", description="2L", priority="high")
        task_store.update("t1", status="done")
        reloaded = TaskStore(tasks_file=tmp_path / "tasks.json")
        tasks = reloaded.list()
        assert len(tasks) == 1
        assert tasks[0].title == "Buy milk"
        assert tasks[0].status == "done"
        assert tasks[0].priority == "high"

    def test_corrupt_file_starts_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "tasks.json"
        path.write_text("{not valid json", encoding="utf-8")
        store = TaskStore(tasks_file=path)
        assert store.list() == []

    def test_non_dict_payload_starts_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "tasks.json"
        path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        store = TaskStore(tasks_file=path)
        assert store.list() == []

    def test_malformed_entries_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "tasks.json"
        payload = {"tasks": [{"title": "no id"}, "junk", {"id": "t9", "title": "ok"}]}
        path.write_text(json.dumps(payload), encoding="utf-8")
        store = TaskStore(tasks_file=path)
        assert [t.id for t in store.list()] == ["t9"]

    def test_next_id_continues_from_existing(self, tmp_path: Path) -> None:
        path = tmp_path / "tasks.json"
        path.write_text(
            json.dumps({"tasks": [{"id": "t5", "title": "existing"}]}),
            encoding="utf-8",
        )
        store = TaskStore(tasks_file=path)
        assert store.create("new").id == "t6"
