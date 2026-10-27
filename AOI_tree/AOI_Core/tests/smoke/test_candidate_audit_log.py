"""AC-074：候选 decision/activate/rollback/recover 结构化 OperationLog；TEMP SQLite，替身 CandidateRelease。"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.api import routes_candidates as rc
from backend.db import database
from backend.db.models import Base, OperationLog
from backend.self_learning.candidate_release import CandidateReleaseError


class CandidateAuditLogTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="aoi_cand_audit_")
        self.addCleanup(tmp.cleanup)
        db = create_engine("sqlite:///" + (Path(tmp.name) / "a.db").as_posix(),
                           connect_args={"check_same_thread": False})
        self.addCleanup(db.dispose)
        Base.metadata.create_all(db)
        self.rel = Mock()
        for p in (patch.object(database, "_engine", db),
                  patch.object(database, "_SessionLocal", sessionmaker(bind=db, expire_on_commit=False)),
                  patch.object(rc, "release", return_value=self.rel)):
            p.start()
            self.addCleanup(p.stop)

    def logs(self):
        with database.session_scope() as s:
            return [(r.action, r.detail, r.extra) for r in s.query(OperationLog).order_by(OperationLog.id)]

    def test_success_paths(self):
        self.rel.decide.return_value = {"state": "approved"}
        self.rel.switch.side_effect = [{"state": "active", "model_mirror": {}},
                                       {"state": "active", "idempotent": True},
                                       {"state": "rolled_back"}]
        self.rel.recover.return_value = {"recovered": [], "model_mirror": {}}
        rc.decision("cat", 3, rc.DecisionRequest(approved=True, report_hash="h1", reason="ok"))
        rc.activate("cat", 3)
        rc.activate("cat", 3)
        rc.rollback("cat", 2)
        rc.recover("cat")
        logs = self.logs()
        self.assertEqual([l[0] for l in logs], ["candidate_decision", "candidate_activate",
                                                "candidate_activate", "candidate_rollback",
                                                "candidate_recover"])
        self.assertEqual(logs[0][1], "category=cat version=3")
        self.assertEqual(logs[0][2], {"category": "cat", "version": 3, "approved": True,
                                      "report_hash": "h1", "reason": "ok", "ok": True,
                                      "state": "approved", "idempotent": False})
        self.assertTrue(logs[2][2]["idempotent"])
        self.assertEqual(logs[3][2]["state"], "rolled_back")
        self.assertEqual(logs[4][2], {"category": "cat", "ok": True, "state": None,
                                      "idempotent": False})

    def test_rejected_logged_with_reason_code(self):
        err = CandidateReleaseError("stale", reason_code="report_stale")
        self.rel.switch.side_effect = err
        with self.assertRaises(HTTPException) as ctx:
            rc.activate("cat", 3)
        self.assertEqual(ctx.exception.status_code, 409)
        (action, _, extra), = self.logs()
        self.assertEqual(action, "candidate_activate_rejected")
        self.assertEqual((extra["ok"], extra["reason_code"]), (False, "report_stale"))
        self.rel.decide.side_effect = ValueError("bad")
        with self.assertRaises(HTTPException):
            rc.decision("cat", 3, rc.DecisionRequest(approved=False, report_hash="h", reason="r"))
        self.assertEqual(self.logs()[-1][0], "candidate_decision_rejected")
        self.assertIsNone(self.logs()[-1][2]["reason_code"])

    def test_audit_failure_does_not_mask_result(self):
        self.rel.switch.return_value = {"state": "active"}
        with patch.object(rc, "log_action", side_effect=RuntimeError("db down")):
            out = rc.activate("cat", 3)
        # round3：依赖锁 warn 模式可附带 dependency_lock_warning，不影响结果
        self.assertEqual(out.get("state"), "active")


if __name__ == "__main__":
    unittest.main(verbosity=2)
