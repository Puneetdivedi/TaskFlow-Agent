"""Tests for the one-time legacy JSON → SQLite migration."""

from __future__ import annotations

import json
from pathlib import Path

from src.memory.file_index import FileIndex
from src.memory.migration import migrate_legacy_data
from src.memory.session_store import SessionStore
from src.memory.task_store import TaskStore


class TestMigrateTasks:
    def test_imports_legacy_tasks(self, tmp_path: Path) -> None:
        legacy = tmp_path / "tasks.json"
        legacy.write_text(
            json.dumps(
                {
                    "tasks": [
                        {"id": "t5", "title": "existing", "priority": "high"},
                        {"id": "t7", "title": "second"},
                    ]
                }
            ),
            encoding="utf-8",
        )
        db = tmp_path / "taskflow.db"
        migrate_legacy_data(db, legacy_tasks=legacy)

        store = TaskStore(db_path=db)
        assert store.get("t5").title == "existing"
        assert store.get("t5").priority == "high"
        assert store.get("t7").title == "second"

    def test_old_json_without_new_fields_gets_defaults(self, tmp_path: Path) -> None:
        legacy = tmp_path / "tasks.json"
        legacy.write_text(
            json.dumps({"tasks": [{"id": "t1", "title": "x"}]}),
            encoding="utf-8",
        )
        db = tmp_path / "taskflow.db"
        migrate_legacy_data(db, legacy_tasks=legacy)

        task = TaskStore(db_path=db).get("t1")
        assert task.due_at == ""
        assert task.every_days == 0
        assert task.status == "todo"

    def test_malformed_entries_skipped(self, tmp_path: Path) -> None:
        legacy = tmp_path / "tasks.json"
        legacy.write_text(
            json.dumps({"tasks": [{"title": "no id"}, "junk", {"id": "t9", "title": "ok"}]}),
            encoding="utf-8",
        )
        db = tmp_path / "taskflow.db"
        migrate_legacy_data(db, legacy_tasks=legacy)

        store = TaskStore(db_path=db)
        assert [t.id for t in store.list()] == ["t9"]

    def test_missing_legacy_tasks_is_noop(self, tmp_path: Path) -> None:
        db = tmp_path / "taskflow.db"
        migrate_legacy_data(db, legacy_tasks=tmp_path / "absent.json")
        assert TaskStore(db_path=db).list() == []

    def test_non_dict_payload_skipped(self, tmp_path: Path) -> None:
        legacy = tmp_path / "tasks.json"
        legacy.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        db = tmp_path / "taskflow.db"
        migrate_legacy_data(db, legacy_tasks=legacy)
        assert TaskStore(db_path=db).list() == []

    def test_corrupt_legacy_tasks_skipped(self, tmp_path: Path) -> None:
        legacy = tmp_path / "tasks.json"
        legacy.write_text("{not valid json", encoding="utf-8")
        db = tmp_path / "taskflow.db"
        migrate_legacy_data(db, legacy_tasks=legacy)  # must not raise
        assert TaskStore(db_path=db).list() == []


class TestMigrateSessions:
    def test_imports_legacy_sessions(self, tmp_path: Path) -> None:
        sess_dir = tmp_path / "sessions"
        sess_dir.mkdir()
        (sess_dir / "work.json").write_text(
            json.dumps(
                {
                    "name": "work",
                    "messages": [{"role": "user", "content": "Hello"}],
                }
            ),
            encoding="utf-8",
        )
        db = tmp_path / "taskflow.db"
        migrate_legacy_data(db, legacy_sessions_dir=sess_dir)

        store = SessionStore(db_path=db)
        assert store.load("work") == [{"role": "user", "content": "Hello"}]

    def test_missing_messages_list_skipped(self, tmp_path: Path) -> None:
        sess_dir = tmp_path / "sessions"
        sess_dir.mkdir()
        (sess_dir / "broken.json").write_text(json.dumps({"name": "broken"}), encoding="utf-8")
        db = tmp_path / "taskflow.db"
        migrate_legacy_data(db, legacy_sessions_dir=sess_dir)

        store = SessionStore(db_path=db)
        assert store.list() == []

    def test_corrupt_session_file_skipped(self, tmp_path: Path) -> None:
        sess_dir = tmp_path / "sessions"
        sess_dir.mkdir()
        (sess_dir / "bad.json").write_text("not json", encoding="utf-8")
        db = tmp_path / "taskflow.db"
        migrate_legacy_data(db, legacy_sessions_dir=sess_dir)  # must not raise
        assert SessionStore(db_path=db).list() == []


class TestMigrateIndex:
    def test_imports_legacy_index(self, tmp_path: Path) -> None:
        legacy = tmp_path / "file_index.json"
        legacy.write_text(
            json.dumps(
                {
                    str(tmp_path): {
                        "file_count": 3,
                        "dir_count": 1,
                        "total_size": 100,
                        "indexed_at": "2026-01-01T00:00:00",
                        "entries": {"a.txt": {"type": "file", "size": 100}},
                    }
                }
            ),
            encoding="utf-8",
        )
        db = tmp_path / "taskflow.db"
        migrate_legacy_data(db, legacy_index=legacy)

        result = FileIndex(db_path=db).query()
        assert "Indexed roots" in result
        assert str(tmp_path) in result


class TestMigrateIdempotency:
    def test_running_twice_does_not_duplicate(self, tmp_path: Path) -> None:
        legacy = tmp_path / "tasks.json"
        legacy.write_text(
            json.dumps({"tasks": [{"id": "t5", "title": "existing"}]}),
            encoding="utf-8",
        )
        db = tmp_path / "taskflow.db"
        migrate_legacy_data(db, legacy_tasks=legacy)
        migrate_legacy_data(db, legacy_tasks=legacy)

        store = TaskStore(db_path=db)
        assert [t.id for t in store.list()] == ["t5"]

    def test_skips_import_when_tasks_exist(self, tmp_path: Path) -> None:
        legacy = tmp_path / "tasks.json"
        legacy.write_text(
            json.dumps({"tasks": [{"id": "t5", "title": "legacy"}]}),
            encoding="utf-8",
        )
        db = tmp_path / "taskflow.db"
        TaskStore(db_path=db).create("fresh")

        # Migration only fills empty tables — existing data is never overwritten.
        migrate_legacy_data(db, legacy_tasks=legacy)
        assert [t.id for t in TaskStore(db_path=db).list()] == ["t1"]

    def test_no_sources_is_noop(self, tmp_path: Path) -> None:
        db = tmp_path / "taskflow.db"
        migrate_legacy_data(db)  # defaults point at ~/.taskflow; may or may not exist
        # Must never raise, and should not create anything in a tmp db.
        assert TaskStore(db_path=db).list() == []
