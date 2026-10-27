"""TEMP真实快照/SQLite发布协议；预测替身，不是算法精度验收。"""
import json
import time
from pathlib import Path
from unittest.mock import patch
from tests.smoke.test_isolated_candidate_path import IsolatedCandidateTest
from backend.self_learning.candidate_release import CandidateRelease


class CandidateReleaseTest(IsolatedCandidateTest):
    # 复用真实TEMP构造，不重复执行父类的测试。
    def setUp(self):
        super().setUp()
        self.run_candidate()
        self.release = CandidateRelease(self.engine)
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from backend.db import database
        from backend.db.models import Base, Model
        db = create_engine("sqlite:///" + str(self.root / "business.sqlite3"))
        self.addCleanup(db.dispose)
        Base.metadata.create_all(db)
        self.sessions = sessionmaker(bind=db, expire_on_commit=False)
        p = patch.object(database, "get_session", self.sessions)
        p.start()
        self.addCleanup(p.stop)
        with self.sessions.begin() as s:
            s.add_all([Model(name="parent", category="test", version="v1",
                             path=str(self.root / "v1"), format="engine_snapshot", is_active=True),
                       Model(name="candidate", category="test", version="v2",
                             path=str(self.root / "v2"), format="engine_snapshot"),
                       Model(name="other", category="other", version="v9",
                             path="other/v9", format="engine_snapshot", is_active=True)])
        self.domains = {}
        for domain in ("old", "new"):
            self.domains[domain] = []
            for label in (0, 1):
                p = self.root / (domain + str(label) + ".png")
                p.write_bytes(bytes([label]))
                self.domains[domain].append({"id": len(self.domains[domain]), "path": str(p), "label": label})
        self.policy = {"max_fpr": 0, "min_recall": 1, "max_gray": 0,
                       "old_recall_drop": 0, "max_predict_ms": 1000, "valid_seconds": 300}

    def attach_candidate_feedback(self, *, invalidated=False, revision=0):
        from backend.db.models import Detection, Feedback
        with self.sessions.begin() as s:
            det = Detection(target_type="image", image_path=self.image,
                            category="test", final_score=0.5, is_anomaly=False,
                            latency_ms=1.0, n_tiles={"region_review": {"revision": revision}})
            s.add(det)
            s.flush()
            fb = Feedback(detection_id=det.id, feedback_type="false_negative",
                          operator_label=1, invalidated=invalidated,
                          learning_trace={"candidate_only": True,
                                          "candidate_guard": {
                                              "feedback_id": 1, "detection_id": det.id,
                                              "invalidated": False,
                                              "region_revision": revision}})
            s.add(fb)
            s.flush()
            fb.learning_trace["candidate_guard"]["feedback_id"] = fb.id
            key, intent = self.release.intent("test", 2)
            result = intent["result"]
            result["event_id"] = f"feedback:{fb.id}"
            with self.release.connection() as conn:
                row = conn.execute("SELECT result FROM candidate_events WHERE event_key=?", (key,)).fetchone()
                stored = json.loads(row[0])
                stored["result"] = result
                conn.execute("UPDATE candidate_events SET result=? WHERE event_key=?", (json.dumps(stored), key))
            fb.learning_trace = {"candidate_only": True, "candidate_guard": {
                "feedback_id": fb.id, "detection_id": det.id,
                "invalidated": False, "region_revision": revision}}

    def evaluate(self):
        from algo.eval.offline import Pipeline
        def predict(pipe, path):
            label = Path(path).read_bytes()[0]
            return {"fused": float(label), "decision": "anomaly" if label else "normal"}
        with patch.object(Pipeline, "predict", predict):
            return self.release.evaluate("test", 2, self.domains, self.policy, 99)

    def approve(self):
        r = self.evaluate()
        self.release.decide("test", 2, True, r["report_hash"], "admin", "fixture approval")
        return r

    def assert_mirror(self, version):
        from backend.db.models import Model
        with self.sessions() as s:
            active = s.query(Model).filter(Model.category == "test", Model.is_active.is_(True)).all()
            self.assertEqual([r.version for r in active], [f"v{version}"])
            self.assertEqual(active[0].format, "engine_snapshot")
            self.assertEqual(s.query(Model).filter(Model.category == "other").one().version, "v9")
            self.assertTrue(s.query(Model).filter(Model.category == "other").one().is_active)

    def test_model_commit_failure_explicit_recovery_and_retry(self):
        from sqlalchemy.orm import Session
        self.approve()
        with patch.object(Session, "commit", side_effect=RuntimeError("business commit injected")):
            with self.assertRaisesRegex(ValueError, "实际版本v2.*镜像同步失败"):
                self.engine.activate("test", 2)
        self.assertEqual(self.engine.current_version("test"), 2)
        self.assertEqual(self.release.status("test", 2)["state"], "activated")
        self.assert_mirror(1)
        self.restart()
        self.release = CandidateRelease(self.engine)
        result = self.release.recover("test")
        self.assertEqual(result["recovered"], 0)
        self.assertEqual(result["model_mirror"]["version"], 2)
        self.assert_mirror(2)
        count = len(self.release.status("test", 2)["events"])
        self.assertTrue(self.engine.activate("test", 2)["idempotent"])
        self.assertEqual(len(self.release.status("test", 2)["events"]), count)
        self.assert_mirror(2)

    def test_model_rollback_sync_failure_and_idempotent_recovery(self):
        from sqlalchemy.orm import Session
        self.approve()
        self.engine.activate("test", 2)
        self.assert_mirror(2)
        with patch.object(Session, "commit", side_effect=RuntimeError("rollback commit injected")):
            with self.assertRaisesRegex(ValueError, "实际版本v1.*镜像同步失败"):
                self.release.switch("test", 1, rollback=True)
        self.assertEqual(self.engine.current_version("test"), 1)
        self.assertEqual(self.release.status("test", 2)["state"], "rolled_back")
        self.assert_mirror(2)
        self.assertTrue(self.release.switch("test", 1, rollback=True)["idempotent"])
        self.assert_mirror(1)
        self.release.recover("test")
        self.assert_mirror(1)

    def test_committed_crash_and_missing_historical_model_rows(self):
        from backend.db.models import Model
        with self.sessions.begin() as s:
            s.query(Model).filter(Model.category == "test").delete()
        self.approve()
        def crash(point):
            if point == "release_committed":
                raise RuntimeError("committed crash injected")
        with patch.object(self.engine, "_candidate_checkpoint", side_effect=crash):
            with self.assertRaises(RuntimeError):
                self.engine.activate("test", 2)
        with self.sessions() as s:
            self.assertEqual(s.query(Model).filter(Model.category == "test").count(), 0)
        self.release.recover("test")
        self.release.recover("test")
        self.assert_mirror(2)
        with self.sessions() as s:
            target = s.query(Model).filter(Model.category == "test").one()
            self.assertEqual(target.path, str(self.root / "v2"))
            self.assertNotIn("dataset_ids", target.metrics)
        self.engine.rollback("test", 1)
        self.assert_mirror(1)
        with self.sessions() as s:
            self.assertEqual(s.query(Model).filter(Model.category == "test").count(), 2)

    def test_mirror_not_written_before_release_commit_or_on_foreign_pointer(self):
        from algo.persist import set_current_version
        self.approve()
        def crash(point):
            if point == "release_pointer":
                raise RuntimeError("pointer crash injected")
        with patch.object(self.engine, "_candidate_checkpoint", side_effect=crash):
            with self.assertRaises(RuntimeError):
                self.engine.activate("test", 2)
        self.assert_mirror(1)
        self.release.recover("test")
        self.assert_mirror(2)
        set_current_version(str(self.root), 7)
        with self.assertRaisesRegex(ValueError, "实际指针一致"):
            self.release.recover("test")
        self.assert_mirror(2)

    def test_release_sqlite_commit_failure_recovery_only_then_mirror(self):
        import sqlite3
        self.approve()
        with self.release.connection() as c:
            c.execute("CREATE TRIGGER fail_commit BEFORE UPDATE OF state ON release_events WHEN NEW.state='committed' BEGIN SELECT RAISE(ABORT, 'release commit injected'); END")
        with self.assertRaisesRegex(sqlite3.IntegrityError, "release commit injected"):
            self.engine.activate("test", 2)
        self.assertEqual(self.engine.current_version("test"), 2)
        self.assertEqual(self.release.status("test", 2)["state"], "approved")
        self.assertEqual(self.release.status("test", 2)["events"][-1]["state"], "prepared")
        self.assert_mirror(1)
        with self.release.connection() as c:
            c.execute("DROP TRIGGER fail_commit")
        self.release.recover("test")
        self.assert_mirror(2)

    def test_duplicate_model_rows_service_does_not_override_unique_mirror(self):
        from backend.db.models import Model
        from backend.self_learning import service
        with self.sessions.begin() as s:
            duplicate = Model(name="duplicate", category="test", version="v2",
                              path=str(self.root / "v2"), format="engine_snapshot", is_active=True)
            s.add(duplicate)
            s.flush()
            duplicate_id = duplicate.id
        self.approve()
        with patch.object(service, "get_engine", return_value=self.engine):
            result = service.activate_model(duplicate_id)
            self.assertNotEqual(result["model_id"], duplicate_id)
            service.activate_model(duplicate_id)
        self.assert_mirror(2)
        self.engine.rollback("test", 1)
        self.assertEqual(self.engine.rollback("test", 1), 1)
        self.assert_mirror(1)

    def test_model_sync_does_not_hold_inference_lock(self):
        import threading
        from sqlalchemy.orm import Session
        self.approve()
        original = Session.commit
        def commit(session):
            acquired = []
            def probe():
                lock = self.engine._pipe_lock("test")
                ok = lock.acquire(timeout=0.5)
                acquired.append(ok)
                if ok:
                    lock.release()
            worker = threading.Thread(target=probe)
            worker.start()
            worker.join(2)
            self.assertEqual(acquired, [True])
            return original(session)
        with patch.object(Session, "commit", commit):
            self.engine.activate("test", 2)
        self.assert_mirror(2)

    def test_replay_evidence_is_traceable_and_memory_only(self):
        evidence = self.engine._replay_evidence(
            "test", 1, {"mode": "smoke", "limit": 1},
            [(self.image, 1)], [(self.image, 0)])
        self.assertTrue(evidence["replay_run_id"].startswith("replay-"))
        self.assertEqual(evidence["parent_version"], 1)
        self.assertEqual(evidence["wal_watermark"], self.engine._journal_mark("test"))
        self.assertEqual(evidence["input_summary"]["count"], 2)
        conn = self.engine._feedback_receipts()
        try:
            self.assertEqual(conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='replay_evidence'").fetchall(), [])
        finally:
            conn.close()

    def test_replay_evidence_rejects_parent_or_wal_mismatch(self):
        evidence = self.engine._replay_evidence("test", 1, {"mode": "smoke"},
                                                [(self.image, 1)], [])
        with self.assertRaisesRegex(ValueError, "父版本不匹配"):
            self.release._validate_replay_evidence("test", 2, evidence)
        self.engine._append_journal("test", {"base": 1, "path": self.image})
        with self.assertRaisesRegex(ValueError, "WAL水位不匹配"):
            self.release._validate_replay_evidence("test", 1, evidence)

    def test_release_requires_explicit_approval(self):
        current = (self.root / "current.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "验收"):
            self.engine.activate("test", 2)
        r = self.evaluate()
        self.assertEqual(r["acceptance"], "passed")
        self.assertEqual(r["data_cutoff"], 99)
        self.assertEqual(r["parent_version"], 1)
        with self.assertRaisesRegex(ValueError, "approved"):
            self.engine.activate("test", 2)
        self.assertEqual((self.root / "current.json").read_bytes(), current)
        self.assert_production()

    def test_unavailable_and_rejection_do_not_touch_production(self):
        r = self.release.evaluate("test", 2, {"old": [], "new": []}, {}, 99)
        self.assertEqual(r["acceptance"], "unavailable")
        self.assertEqual(r["domains"], {})
        with self.assertRaises(ValueError):
            self.release.decide("test", 2, True, r["report_hash"], "admin", "no data")
        self.release.decide("test", 2, False, r["report_hash"], "admin", "no data")
        with self.assertRaises(ValueError):
            self.engine.activate("test", 2)
        self.assert_production()

    def test_duplicate_conflict_and_restart(self):
        r = self.approve()
        out = self.release.decide("test", 2, True, r["report_hash"], "admin", "fixture approval")
        self.assertTrue(out["idempotent"])
        with self.assertRaisesRegex(ValueError, "冲突"):
            self.release.decide("test", 2, False, r["report_hash"], "admin", "changed")
        self.restart()
        self.release = CandidateRelease(self.engine)
        self.assertEqual(self.release.status("test", 2)["state"], "approved")
        self.engine.activate("test", 2)
        self.assertEqual(self.release.status("test", 2)["state"], "activated")
        from algo.eval.offline import Pipeline
        with patch.object(Pipeline, "predict", return_value={"fused": 0.2, "decision": "normal"}):
            self.assertEqual(self.engine.predict_image_path("test", self.image).model_version, 2)
        self.assertEqual(self.engine.rollback("test", 1), 1)
        self.assertEqual(self.engine.current_version("test"), 1)
        status = self.release.status("test", 2)
        self.assertEqual(status["state"], "rolled_back")
        event = status["events"][-1]
        self.assertEqual(event["payload"]["target_version"], 1)
        self.assertIn("event_watermark", event["payload"])
        with self.assertRaises(ValueError):
            self.engine.activate("test", 2)

    def test_parent_change_and_expiration_fail_closed(self):
        self.approve()
        from algo.persist import set_current_version
        set_current_version(str(self.root), 8)
        with self.assertRaisesRegex(ValueError, "父版本"):
            self.engine.activate("test", 2)
        self.assertEqual(self.engine.current_version("test"), 8)
        set_current_version(str(self.root), 1)
        with patch("backend.self_learning.candidate_release.time.time", return_value=time.time() + 1000):
            with self.assertRaisesRegex(ValueError, "过期"):
                self.engine.activate("test", 2)
        self.assert_production()

    def test_data_wal_and_snapshot_changes_fail_closed(self):
        self.approve()
        p = Path(self.domains["old"][0]["path"])
        original = p.read_bytes()
        p.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "数据"):
            self.engine.activate("test", 2)
        p.write_bytes(original)
        wp = Path(self.engine._journal_path("test"))
        original = wp.read_bytes() if wp.exists() else b""
        wp.write_bytes(original + b"{}\n")
        with self.assertRaisesRegex(ValueError, "WAL"):
            self.engine.activate("test", 2)
        wp.write_bytes(original)
        meta = self.root / "v2" / "meta.json"
        original = meta.read_bytes()
        meta.write_bytes(original + b" ")
        with self.assertRaisesRegex(ValueError, "完整性"):
            self.release.switch("test", 2)
        self.assert_production()

    def test_prepared_abort_and_rollback_recovery(self):
        self.approve()
        def crash(point):
            if point == "release_prepared":
                raise RuntimeError("injected crash")
        with patch.object(self.engine, "_candidate_checkpoint", side_effect=crash):
            with self.assertRaises(RuntimeError):
                self.engine.activate("test", 2)
        self.assert_production()
        with self.assertRaisesRegex(ValueError, "恢复"):
            self.engine.activate("test", 2)
        self.release.recover("test")
        self.assertEqual(self.release.status("test", 2)["events"][-1]["state"], "aborted")
        self.engine.activate("test", 2)
        def crash_pointer(point):
            if point == "release_pointer":
                raise RuntimeError("injected crash")
        with patch.object(self.engine, "_candidate_checkpoint", side_effect=crash_pointer):
            with self.assertRaises(RuntimeError):
                self.engine.rollback("test", 1)
        self.restart()
        self.release = CandidateRelease(self.engine)
        self.release.recover("test")
        self.assertEqual(self.release.status("test", 2)["state"], "rolled_back")
        self.assertEqual(self.engine.current_version("test"), 1)

    def test_prepared_discovery_is_read_only_and_reports_consistency(self):
        import sqlite3
        db_path = self.engine.storage_dir + "/feedback_receipts.sqlite3"
        with self.release.connection() as c:
            c.execute("INSERT INTO release_events(category,candidate_id,action,state,payload) VALUES (?,?,?,?,?)",
                      ("test", "candidate-1", "activate", "prepared",
                       '{"from_version": 1, "target_version": 2, "event_watermark": {"sha256": "bad", "bytes": 0, "records": 0}}'))
        before = (self.engine.current_version("test"), Path(self.engine._journal_path("test")).read_bytes() if Path(self.engine._journal_path("test")).exists() else b"")
        found = self.release.discover_prepared("test")
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["pointer_consistency"], "parent")
        self.assertEqual(found[0]["recommendation"], "manual_required")
        self.assertFalse(found[0]["wal_consistent"])
        self.assertEqual((self.engine.current_version("test"), Path(self.engine._journal_path("test")).read_bytes() if Path(self.engine._journal_path("test")).exists() else b""), before)
        c = sqlite3.connect(db_path)
        try:
            self.assertEqual(c.execute("SELECT state FROM release_events WHERE candidate_id='candidate-1'").fetchone()[0], "prepared")
        finally:
            c.close()

    def test_prepared_discovery_handles_missing_historical_database(self):
        import os
        path = os.path.join(self.engine.storage_dir, "feedback_receipts.sqlite3")
        if os.path.exists(path):
            os.remove(path)
        self.assertEqual(self.release.discover_prepared("test"), [])
        self.assertFalse(os.path.exists(path))

    def test_prepared_discovery_http_engineer_only(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from backend.api import routes_candidates
        app = FastAPI()
        @app.middleware("http")
        async def role(request, call_next):
            request.state.role = request.headers.get("X-Test-Role", "operator")
            return await call_next(request)
        app.include_router(routes_candidates.router, prefix="/api")
        with patch.object(routes_candidates, "get_engine", return_value=self.engine), TestClient(app) as client:
            self.assertEqual(client.get("/api/candidates/recovery/prepared").status_code, 403)
            response = client.get("/api/candidates/recovery/prepared", headers={"X-Test-Role": "engineer"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), [])

    def test_minimal_http_role_boundary(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from backend.api import routes_candidates
        app = FastAPI()
        @app.middleware("http")
        async def role(request, call_next):
            request.state.role = request.headers.get("X-Test-Role", "operator")
            return await call_next(request)
        app.include_router(routes_candidates.router, prefix="/api")
        with patch.object(routes_candidates, "get_engine", return_value=self.engine), TestClient(app) as client:
            self.assertEqual(client.get("/api/candidates/test/2").status_code, 403)
            self.assertEqual(client.get("/api/candidates/test/2", headers={"X-Test-Role": "engineer"}).status_code, 200)
            self.assertEqual(client.post("/api/candidates/test/2/activate", headers={"X-Test-Role": "engineer"}).status_code, 403)
            self.assertEqual(client.post("/api/candidates/test/2/activate", headers={"X-Test-Role": "admin"}).status_code, 409)
        self.assert_production()

    def test_release_pointer_crash_restart_recovery(self):
        self.approve()
        def checkpoint(point):
            if point == "release_pointer":
                raise RuntimeError("injected crash")
        with patch.object(self.engine, "_candidate_checkpoint", side_effect=checkpoint):
            with self.assertRaises(RuntimeError):
                self.engine.activate("test", 2)
        self.restart()
        self.release = CandidateRelease(self.engine)
        self.assertEqual(self.release.status("test", 2)["state"], "approved")
        self.release.recover("test")
        self.assertEqual(self.release.status("test", 2)["state"], "activated")
        self.assertEqual(self.engine.current_version("test"), 2)
        self.assertEqual(self.release.recover("test")["recovered"], 0)

    def test_candidate_feedback_invalidated_after_training_blocks_approval_and_activation(self):
        self.attach_candidate_feedback()
        report = self.evaluate()
        from backend.db.models import Feedback
        with self.sessions.begin() as s:
            s.query(Feedback).update({Feedback.invalidated: True})
        with self.assertRaisesRegex(ValueError, "作废") as caught:
            self.release.decide("test", 2, True, report["report_hash"], "admin", "fixture approval")
        self.assertEqual(caught.exception.reason_code, "PARENT_REBUILD_REQUIRED_INVALIDATED_FEEDBACK")
        self.assertEqual(self.release.status("test", 2)["state"], "blocked")
        self.assertEqual(self.engine.current_version("test"), 1)

    def test_candidate_detection_revision_after_training_blocks_approval(self):
        self.attach_candidate_feedback(revision=0)
        report = self.evaluate()
        from backend.db.models import Detection
        with self.sessions.begin() as s:
            det = s.query(Detection).one()
            det.n_tiles = {"region_review": {"revision": 1}}
        with self.assertRaisesRegex(ValueError, "人工修订") as caught:
            self.release.decide("test", 2, True, report["report_hash"], "admin", "fixture approval")
        self.assertEqual(caught.exception.reason_code, "PARENT_REBUILD_REQUIRED_REVIEW_REVISION")
        self.assertEqual(self.release.status("test", 2)["state"], "blocked")
        self.assertEqual(self.engine.current_version("test"), 1)

    def test_candidate_feedback_invalidated_after_approval_blocks_activation(self):
        self.attach_candidate_feedback()
        report = self.evaluate()
        self.release.decide("test", 2, True, report["report_hash"], "admin", "fixture approval")
        from backend.db.models import Feedback
        with self.sessions.begin() as s:
            s.query(Feedback).update({Feedback.invalidated: True})
        with self.assertRaisesRegex(ValueError, "作废"):
            self.engine.activate("test", 2)
        self.assertEqual(self.engine.current_version("test"), 1)

    def test_candidate_detection_revision_after_approval_blocks_activation(self):
        self.attach_candidate_feedback(revision=0)
        report = self.evaluate()
        self.release.decide("test", 2, True, report["report_hash"], "admin", "fixture approval")
        from backend.db.models import Detection
        with self.sessions.begin() as s:
            det = s.query(Detection).one()
            det.n_tiles = {"region_review": {"revision": 1}}
        with self.assertRaisesRegex(ValueError, "人工修订"):
            self.engine.activate("test", 2)
        self.assertEqual(self.engine.current_version("test"), 1)

    def test_status_release_events_are_isolated_between_candidates(self):
        with self.release.connection() as conn:
            conn.execute("INSERT INTO candidate_events(event_key,payload,parent,state,result) VALUES (?,?,?,?,?)",
                         ("candidate:two", "{}", 1, "completed",
                          '{"result":{"category":"test","version_num":3}}'))
            conn.execute("INSERT INTO release_events(category,candidate_id,action,state,payload) VALUES (?,?,?,?,?)",
                         ("test", "candidate:one", "approve", "approved", "{}"))
            conn.execute("INSERT INTO release_events(category,candidate_id,action,state,payload) VALUES (?,?,?,?,?)",
                         ("test", "candidate:two", "approve", "approved", "{}"))
        first = self.release.status("test", 2)
        second = self.release.status("test", 3)
        self.assertEqual([event["candidate_id"] for event in first["events"]], ["candidate:one"])
        self.assertEqual([event["candidate_id"] for event in second["events"]], ["candidate:two"])

    def test_status_historical_missing_candidate_id_fails_closed(self):
        with self.release.connection() as conn:
            conn.execute("INSERT INTO release_events(category,candidate_id,action,state,payload) VALUES (?,?,?,?,?)",
                         ("test", None, "activate", "committed", '{"candidate_event":"candidate:one"}'))
            conn.execute("INSERT INTO release_events(category,candidate_id,action,state,payload) VALUES (?,?,?,?,?)",
                         ("test", None, "activate", "committed", '{"target_version":999}'))
        events = self.release.status("test", 2)["events"]
        self.assertEqual(events[0]["candidate_id"], "candidate:one")
        self.assertEqual(events[1]["candidate_id"], None)
        self.assertEqual(events[1]["state"], "orphan")
        self.assertEqual(events[1]["availability"], "unavailable")

    def test_acceptance_report_keeps_stable_failure_details(self):
        report = self.release.evaluate("test", 2, {"old": [], "new": []}, {}, 99)
        self.assertEqual(report["acceptance"], "unavailable")
        self.assertEqual(report["reason_code"], "CANDIDATE_ACCEPTANCE_UNAVAILABLE")
        self.assertEqual(report["violations"], [])
        self.assertFalse(report["retryable"])

    def test_candidate_build_failure_details_are_persisted(self):
        from backend.self_learning.candidate_release import failure_details
        details = failure_details(ValueError("snapshot missing"), "build")
        self.assertEqual(details["reason_code"], "CANDIDATE_BUILD_FAILED")
        self.assertEqual(details["violations"][0]["code"], "CANDIDATE_BUILD_FAILED")
        self.assertFalse(details["retryable"])


# 不继承执行旧case，但保留fixture辅助函数。
for _name in list(IsolatedCandidateTest.__dict__):
    if _name.startswith("test_") and _name not in CandidateReleaseTest.__dict__:
        setattr(CandidateReleaseTest, _name, None)
