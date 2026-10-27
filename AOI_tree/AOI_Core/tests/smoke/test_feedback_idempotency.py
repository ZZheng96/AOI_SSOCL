"""SS-B 小型测试：TEMP receipt/WAL/业务库，不访问生产资产。"""
import os
import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.engine import DetectionEngine
from backend.db import database
from backend.db.models import Base, Detection, Feedback, Image, StatsDaily
from backend.self_learning import service


class FeedbackIdempotencyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="aoi_feedback_")
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "image.png")
        with open(self.path, "wb") as f:
            f.write(b"test")
        self.pipe = SimpleNamespace(feedback=Mock(return_value={}), handler=None)
        self.engine = self.make_engine(self.pipe)

    def make_engine(self, pipe):
        e = DetectionEngine.__new__(DetectionEngine)
        e.storage_dir = self.tmp.name
        e.snap_root = os.path.join(self.tmp.name, "snapshots")
        e._ensure_loaded = Mock()
        e._pipe_entry = Mock(return_value=(pipe, 1))
        lock = threading.RLock()
        e._pipe_lock = Mock(return_value=lock)
        e.current_version = Mock(return_value=1)
        e.predict_image_path = Mock(return_value=SimpleNamespace(
            score=0.2, decision="normal", slot_scores={}))
        return e

    def submit(self, **kwargs):
        return self.engine.submit_feedback("cat", self.path, "wrong", label=1, **kwargs)

    def test_duplicate_and_payload_conflict(self):
        self.submit(feedback_id=7)
        self.assertTrue(self.submit(feedback_id=7)["idempotent"])
        with self.assertRaisesRegex(ValueError, "payload"):
            self.engine.submit_feedback("cat", self.path, "correct", label=0, feedback_id=7)
        self.assertEqual(self.pipe.feedback.call_count, 1)

    def test_failure_retry_and_no_id_legacy(self):
        self.pipe.feedback.side_effect = [RuntimeError("learn failed"), {}, {}, {}]
        with self.assertRaises(RuntimeError):
            self.submit(feedback_id=8)
        self.submit(feedback_id=8)
        self.submit()
        self.submit()
        self.assertEqual(self.pipe.feedback.call_count, 4)

    def test_concurrent(self):
        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(lambda _: self.submit(event_id="evt"), range(12)))
        self.assertEqual(self.pipe.feedback.call_count, 1)
        self.assertEqual(sum(bool(r.get("idempotent")) for r in results), 11)

    def test_restart_replay_and_clear_receipt_survives(self):
        self.submit(feedback_id=9)
        recovered = SimpleNamespace(feedback=Mock(return_value={}), handler=None)
        e = self.make_engine(recovered)
        self.assertEqual(e._replay_journal("cat", recovered, 1), 1)
        self.assertTrue(e.submit_feedback("cat", self.path, "wrong", label=1,
                                          feedback_id=9)["idempotent"])
        e._clear_journal("cat")
        self.assertTrue(e.submit_feedback("cat", self.path, "wrong", label=1,
                                          feedback_id=9)["idempotent"])
        self.assertEqual(recovered.feedback.call_count, 1)

    def test_duplicate_wal_and_failure_receipt_gap_recovery(self):
        rec = dict(base=1, path=self.path, verdict="wrong", label=1, box=None,
                   feedback_id=10)
        self.engine._append_journal("cat", rec)
        self.engine._append_journal("cat", rec)
        self.assertEqual(self.engine._replay_journal("cat", self.pipe, 1), 1)
        self.assertTrue(self.submit(feedback_id=10)["idempotent"])
        self.assertEqual(self.pipe.feedback.call_count, 1)

    def test_replay_audit_reports_invalid_json_and_single_failure(self):
        rec = dict(base=1, path=self.path, verdict="wrong", label=1, box=None,
                   feedback_id=14)
        self.engine._append_journal("cat", rec)
        with open(self.engine._journal_path("cat"), "a", encoding="utf-8") as f:
            f.write("{not-json}\\n")
        self.pipe.feedback.side_effect = RuntimeError("replay failed")
        self.assertEqual(self.engine._replay_journal("cat", self.pipe, 1), 0)
        audit = self.engine._last_journal_replay_audit
        self.assertEqual({item["status"] for item in audit},
                         {"invalid_json", "replay_failed"})
        invalid = next(item for item in audit if item["status"] == "invalid_json")
        failed = next(item for item in audit if item["status"] == "replay_failed")
        self.assertEqual(invalid["line"], 2)
        self.assertEqual(failed["error"], "replay failed")

    def test_wal_failure_does_not_complete(self):
        complete = Mock()
        with patch.object(self.engine, "_append_journal", side_effect=OSError("disk")):
            with self.assertRaises(OSError):
                self.submit(feedback_id=11, on_complete=complete)
        complete.assert_not_called()
        conn = self.engine._feedback_receipts()
        self.assertEqual(conn.execute("SELECT count(*) FROM receipts").fetchone()[0], 0)
        conn.close()

    def test_callback_failure_retry_does_not_relearn(self):
        with self.assertLogs("backend.engine", level="ERROR"):
            self.submit(feedback_id=12, on_complete=Mock(side_effect=RuntimeError("db")))
        complete = Mock()
        self.assertTrue(self.submit(feedback_id=12, on_complete=complete)["idempotent"])
        complete.assert_called_once()
        self.assertEqual(self.pipe.feedback.call_count, 1)

    def test_async_duplicate_waits_for_success(self):
        release = threading.Event()
        entered = threading.Event()
        def learn(*args, **kwargs):
            entered.set()
            release.wait(5)
            return {}
        self.pipe.feedback.side_effect = learn
        complete = threading.Event()
        result = self.submit(feedback_id=13, timeout_ms=1, on_complete=complete.set)
        self.assertTrue(result["async"])
        self.assertTrue(entered.wait(2))
        release.set()
        self.assertTrue(complete.wait(2))
        self.assertTrue(self.submit(feedback_id=13)["idempotent"])
        self.assertEqual(self.pipe.feedback.call_count, 1)

    def test_migration_duplicate_history_fails_without_deleting(self):
        from sqlalchemy import text
        db = create_engine("sqlite:///" + os.path.join(self.tmp.name, "legacy.sqlite3"))
        with db.begin() as c:
            c.execute(text("CREATE TABLE feedback (id INTEGER PRIMARY KEY, source_event_id TEXT)"))
            c.execute(text("INSERT INTO feedback VALUES (1, 'same'), (2, 'same')"))
            c.execute(text("PRAGMA user_version=6"))
        with self.assertRaises(Exception):
            database._migrate_sqlite(db)
        with db.begin() as c:
            self.assertEqual(c.execute(text("SELECT count(*) FROM feedback")).scalar(), 2)
            self.assertEqual(c.execute(text("PRAGMA user_version")).scalar(), 6)
        db.dispose()

    def test_service_duplicate_conflict_retry_concurrency(self):
        db = create_engine("sqlite:///" + os.path.join(self.tmp.name, "business.sqlite3"),
                           connect_args={"check_same_thread": False})
        Base.metadata.create_all(db)
        sessions = sessionmaker(bind=db, expire_on_commit=False)
        with sessions.begin() as s:
            s.add(Image(path=self.path, category="cat", split="feedback", label="anomaly"))
            s.add(Detection(id=1, image_path=self.path, category="cat", target_type="review",
                            final_score=0.2, is_anomaly=False, latency_ms=1))
        with patch.object(database, "get_session", sessions), \
             patch.object(service, "get_engine", return_value=self.engine), \
             patch.object(service, "notify_event"), patch.object(service, "log_action"):
            self.pipe.feedback.side_effect = [RuntimeError("unavailable"), {}]
            a = service.submit_feedback_record(1, "false_negative", 1, source_event_id="source",
                                             candidate_only=False, allow_legacy_online=True)
            self.assertIsNone(a["engine_update"])
            b = service.submit_feedback_record(1, "false_negative", 1, source_event_id="source",
                                             candidate_only=False, allow_legacy_online=True)
            self.assertEqual(a["feedback_id"], b["feedback_id"])
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(lambda _: service.submit_feedback_record(
                    1, "false_negative", 1, source_event_id="source",
                    candidate_only=False, allow_legacy_online=True), range(8)))
            with self.assertRaises(service.FeedbackConflict):
                service.submit_feedback_record(1, "confirmed", 0, source_event_id="source",
                                               candidate_only=False, allow_legacy_online=True)
            with sessions() as s:
                self.assertEqual(s.query(Feedback).count(), 1)
                self.assertTrue(s.query(Feedback).one().consumed)
                self.assertEqual(s.query(StatsDaily).one().n_feedback, 1)
        self.assertEqual(self.pipe.feedback.call_count, 2)
        db.dispose()


if __name__ == "__main__":
    unittest.main(verbosity=2)
