"""Tests for the persistent task store."""

from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timedelta
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

    def test_create_persists_db(self, task_store: TaskStore, tmp_path: Path) -> None:
        task_store.create("Buy milk")
        assert (tmp_path / "tasks.db").exists()

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
        reloaded = TaskStore(db_path=tmp_path / "tasks.db")
        tasks = reloaded.list()
        assert len(tasks) == 1
        assert tasks[0].title == "Buy milk"
        assert tasks[0].status == "done"
        assert tasks[0].priority == "high"

    def test_corrupt_db_starts_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "tasks.db"
        path.write_text("this is not a sqlite database", encoding="utf-8")
        store = TaskStore(db_path=path)
        assert store.list() == []


class TestTaskScheduling:
    def test_create_defaults_due_at_and_every_days(self, task_store: TaskStore) -> None:
        task = task_store.create("Buy milk")
        assert task.due_at == ""
        assert task.every_days == 0

    def test_create_accepts_due_at_and_every_days(self, task_store: TaskStore) -> None:
        task = task_store.create("Water plants", due_at="2026-08-10", every_days=7)
        assert task.due_at == "2026-08-10"
        assert task.every_days == 7

    def test_create_rejects_bad_due_at(self, task_store: TaskStore) -> None:
        with pytest.raises(ValueError, match="Invalid due_at"):
            task_store.create("Task", due_at="not-a-date")

    def test_create_rejects_timezone_due_at(self, task_store: TaskStore) -> None:
        with pytest.raises(ValueError, match="timezone"):
            task_store.create("Task", due_at="2026-08-10T09:00+05:00")

    def test_create_rejects_negative_every_days(self, task_store: TaskStore) -> None:
        with pytest.raises(ValueError, match="every_days"):
            task_store.create("Task", every_days=-1)

    def test_create_rejects_bool_every_days(self, task_store: TaskStore) -> None:
        with pytest.raises(ValueError, match="every_days"):
            task_store.create("Task", every_days=True)

    def test_update_due_at(self, task_store: TaskStore) -> None:
        task_store.create("Task")
        updated = task_store.update("t1", due_at="2026-08-10")
        assert updated.due_at == "2026-08-10"

    def test_update_every_days(self, task_store: TaskStore) -> None:
        task_store.create("Task")
        updated = task_store.update("t1", every_days=7)
        assert updated.every_days == 7

    def test_update_invalid_due_at_raises(self, task_store: TaskStore) -> None:
        task_store.create("Task")
        with pytest.raises(ValueError, match="Invalid due_at"):
            task_store.update("t1", due_at="bogus")

    def test_update_negative_every_days_raises(self, task_store: TaskStore) -> None:
        task_store.create("Task")
        with pytest.raises(ValueError, match="every_days"):
            task_store.update("t1", every_days=-1)

    def test_update_clears_due_at_with_empty_string(self, task_store: TaskStore) -> None:
        task_store.create("Task", due_at="2026-08-10")
        updated = task_store.update("t1", due_at="")
        assert updated.due_at == ""

    def test_update_clears_recurrence_with_zero(self, task_store: TaskStore) -> None:
        task_store.create("Task", every_days=7)
        updated = task_store.update("t1", every_days=0)
        assert updated.every_days == 0


class TestTaskComplete:
    def test_complete_non_recurring_marks_done(self, task_store: TaskStore) -> None:
        task_store.create("Buy milk", due_at="2026-08-10")
        completed = task_store.complete("t1")
        assert completed.status == "done"
        assert completed.due_at == "2026-08-10"  # untouched

    def test_complete_recurring_rolls_due_forward(self, task_store: TaskStore) -> None:
        task_store.create("Water plants", due_at="2026-08-10", every_days=7)
        completed = task_store.complete("t1")
        assert completed.status == "todo"
        assert completed.due_at == "2026-08-17"

    def test_complete_recurring_preserves_time_of_day(self, task_store: TaskStore) -> None:
        task_store.create("Standup", due_at="2026-08-10T09:30:00", every_days=1)
        completed = task_store.complete("t1")
        assert completed.due_at == "2026-08-11T09:30:00"

    def test_complete_recurring_uses_interval(self, task_store: TaskStore) -> None:
        task_store.create("Task", due_at="2026-08-10", every_days=2)
        completed = task_store.complete("t1")
        assert completed.due_at == "2026-08-12"

    def test_complete_recurring_without_due_at_marks_done(self, task_store: TaskStore) -> None:
        task_store.create("Task", every_days=7)
        completed = task_store.complete("t1")
        assert completed.status == "done"

    def test_complete_recurring_persists(self, task_store: TaskStore, tmp_path: Path) -> None:
        task_store.create("Water plants", due_at="2026-08-10", every_days=7)
        task_store.complete("t1")
        reloaded = TaskStore(db_path=tmp_path / "tasks.db")
        assert reloaded.get("t1").status == "todo"
        assert reloaded.get("t1").due_at == "2026-08-17"

    def test_complete_missing_raises_key_error(self, task_store: TaskStore) -> None:
        with pytest.raises(KeyError, match="not found"):
            task_store.complete("t99")


class TestTaskPlanAutoRun:
    def test_create_defaults_plan_and_auto_run(self, task_store: TaskStore) -> None:
        task = task_store.create("Buy milk")
        assert task.plan == ""
        assert task.auto_run is False

    def test_create_accepts_plan_and_auto_run(self, task_store: TaskStore) -> None:
        task = task_store.create(
            "Daily report",
            plan="Write README.md then list the directory.",
            auto_run=True,
        )
        assert task.plan == "Write README.md then list the directory."
        assert task.auto_run is True

    def test_update_plan_and_auto_run(self, task_store: TaskStore) -> None:
        task_store.create("Task")
        updated = task_store.update("t1", plan="step one\nstep two", auto_run=True)
        assert updated.plan == "step one\nstep two"
        assert updated.auto_run is True

    def test_update_clears_plan_with_empty_string(self, task_store: TaskStore) -> None:
        task_store.create("Task", plan="old plan")
        updated = task_store.update("t1", plan="")
        assert updated.plan == ""

    def test_update_auto_run_back_to_false(self, task_store: TaskStore) -> None:
        task_store.create("Task", auto_run=True)
        updated = task_store.update("t1", auto_run=False)
        assert updated.auto_run is False

    def test_plan_roundtrips_through_reload(self, task_store: TaskStore, tmp_path: Path) -> None:
        task_store.create("Task", plan="do things", auto_run=True)
        reloaded = TaskStore(db_path=tmp_path / "tasks.db")
        task = reloaded.get("t1")
        assert task.plan == "do things"
        assert task.auto_run is True


class TestTaskAdvance:
    def test_advance_recurring_rolls_due_and_keeps_status(self, task_store: TaskStore) -> None:
        task_store.create("Water plants", due_at="2026-08-10", every_days=7)
        task_store.update("t1", status="in_progress")
        advanced = task_store.advance("t1")
        assert advanced.status == "in_progress"  # status preserved
        assert advanced.due_at == "2026-08-17"

    def test_advance_one_shot_marks_done(self, task_store: TaskStore) -> None:
        task_store.create("Buy milk", due_at="2026-08-10")
        advanced = task_store.advance("t1")
        assert advanced.status == "done"
        assert advanced.due_at == "2026-08-10"  # untouched

    def test_advance_recurring_preserves_time_of_day(self, task_store: TaskStore) -> None:
        task_store.create("Standup", due_at="2026-08-10T09:30:00", every_days=1)
        advanced = task_store.advance("t1")
        assert advanced.due_at == "2026-08-11T09:30:00"

    def test_advance_recurring_persists(self, task_store: TaskStore, tmp_path: Path) -> None:
        task_store.create("Water plants", due_at="2026-08-10", every_days=7)
        task_store.advance("t1")
        reloaded = TaskStore(db_path=tmp_path / "tasks.db")
        assert reloaded.get("t1").due_at == "2026-08-17"
        assert reloaded.get("t1").status == "todo"

    def test_advance_missing_raises_key_error(self, task_store: TaskStore) -> None:
        with pytest.raises(KeyError, match="not found"):
            task_store.advance("t99")


class TestTaskMigration:
    def test_migrates_old_schema_columns(self, tmp_path: Path) -> None:
        """A database created before plan/auto_run gains the columns."""
        db = tmp_path / "legacy.db"
        conn = sqlite3.connect(db)
        conn.execute(
            "CREATE TABLE tasks ("
            "id TEXT PRIMARY KEY, title TEXT NOT NULL, "
            "description TEXT NOT NULL DEFAULT '', status TEXT NOT NULL DEFAULT 'todo', "
            "priority TEXT NOT NULL DEFAULT 'medium', created_at TEXT NOT NULL DEFAULT '', "
            "updated_at TEXT NOT NULL DEFAULT '', due_at TEXT NOT NULL DEFAULT '', "
            "every_days INTEGER NOT NULL DEFAULT 0)"
        )
        conn.execute("INSERT INTO tasks (id, title) VALUES ('t1', 'Old task')")
        conn.commit()
        conn.close()

        store = TaskStore(db_path=db)
        old = store.get("t1")
        assert old.plan == ""
        assert old.auto_run is False

        created = store.create("New task", plan="steps", auto_run=True)
        assert created.plan == "steps"
        assert created.auto_run is True


class TestTaskDue:
    @staticmethod
    def _offset_days(days: int) -> str:
        return (datetime.now() + timedelta(days=days)).date().isoformat()

    def test_due_empty_store(self, task_store: TaskStore) -> None:
        assert task_store.due() == []

    def test_due_returns_overdue_and_due_today(self, task_store: TaskStore) -> None:
        task_store.create("Overdue", due_at=self._offset_days(-1))
        task_store.create("Today", due_at=self._offset_days(0))
        assert {t.title for t in task_store.due()} == {"Overdue", "Today"}

    def test_due_excludes_done_tasks(self, task_store: TaskStore) -> None:
        task_store.create("Done overdue", due_at=self._offset_days(-1))
        task_store.update("t1", status="done")
        assert task_store.due() == []

    def test_due_excludes_no_due_at(self, task_store: TaskStore) -> None:
        task_store.create("No due")
        assert task_store.due() == []

    def test_due_ahead_days_includes_future(self, task_store: TaskStore) -> None:
        task_store.create("Later", due_at=self._offset_days(3))
        assert task_store.due() == []
        assert [t.id for t in task_store.due(ahead_days=7)] == ["t1"]

    def test_due_sorted_soonest_first(self, task_store: TaskStore) -> None:
        now = datetime.now()
        task_store.create("Third", due_at=(now - timedelta(hours=6)).isoformat())
        task_store.create("First", due_at=(now - timedelta(days=2)).isoformat())
        task_store.create("Second", due_at=(now - timedelta(days=1)).isoformat())
        assert [t.title for t in task_store.due()] == ["First", "Second", "Third"]

    def test_due_rejects_negative_ahead_days(self, task_store: TaskStore) -> None:
        with pytest.raises(ValueError, match="ahead_days"):
            task_store.due(ahead_days=-1)
