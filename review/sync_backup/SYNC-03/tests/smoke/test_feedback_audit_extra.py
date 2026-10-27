"""AC-055：反馈相关操作日志结构化 extra；TEMP SQLite 真实落库，仅替身推理引擎。"""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.api import routes_feedback, routes_review
from backend.api.schemas import FeedbackCreateRequest
from backend.db import database
from backend.db.models import Base, Detection, Image, OperationLog
from backend.self_learning import service

_FORBIDDEN = {"learning_trace", "image_path", "comment", "activated", "applied"}


class FeedbackAuditExtraTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="aoi_fb_audit_")
        self.addCleanup(self.tmp.cleanup)
        self.db = create_engine("sqlite:///" + (Path(self.tmp.name) / "audit.db").as_posix(),
                                connect_args={"check_same_thread": False})
        self.addCleanup(self.db.dispose)
        Base.metadata.create_all(self.db)
        # 当前激活版本 8；检测实际推理版本 7（n_tiles.model_version）
        self.engine = SimpleNamespace(
            current_version=Mock(return_value=8),
            predict_image_path=Mock(return_value=SimpleNamespace(
                score=.2, decision="normal", slot_scores={})),
            submit_feedback=Mock(side_effect=AssertionError("排队阶段不得在线学习")))
        for p in (patch.object(database, "_engine", self.db),
                  patch.object(database, "_SessionLocal", sessionmaker(
                      bind=self.db, expire_on_commit=False)),
                  patch.object(service, "get_engine", return_value=self.engine),
                  patch.object(service, "notify_event")):
            p.start()
            self.addCleanup(p.stop)
        with database.session_scope() as s:
            s.add(Image(path="fixture.png", category="cat", split="feedback", label="anomaly"))
            s.add(Detection(id=1, image_path="fixture.png", category="cat", target_type="image",
                            final_score=.2, is_anomaly=False, latency_ms=1,
                            n_tiles={"decision": "normal", "model_version": 7}))
            s.add(Detection(id=2, image_path="fixture.png", category="cat", target_type="image",
                            final_score=.2, is_anomaly=False, latency_ms=1,
                            n_tiles={"decision": "normal"}))  # 旧数据：缺版本
            s.add(Detection(id=3, image_path="fixture.png", category="cat", target_type="pipeline",
                            final_score=.5, is_anomaly=False, latency_ms=1,
                            n_tiles={"decision": "gray", "model_version": 7}))
            s.add(Detection(id=4, image_path="fixture.png", category="cat", target_type="pipeline",
                            final_score=.5, is_anomaly=False, latency_ms=1,
                            n_tiles={"decision": "gray", "review_stage": "tampered"}))

    def logs(self, action=None):
        with database.session_scope() as s:
            q = s.query(OperationLog)
            if action:
                q = q.filter(OperationLog.action == action)
            return [(r.user, r.action, r.detail, r.extra) for r in q.order_by(OperationLog.id)]

    def feedback(self, det_id=1, event="evt-1", **kw):
        return routes_feedback.api_create_feedback(FeedbackCreateRequest(
            detection_id=det_id, feedback_type="false_negative", operator_label=1,
            source_event_id=event, comment="评论不入审计", **kw))

    def assert_clean(self, extra):
        self.assertFalse(_FORBIDDEN & set(extra), extra)

    def test_candidate_queued_extra_and_version(self):
        out = self.feedback(operator="op1")
        self.assertEqual(out["learning_status"], "candidate_queued")
        self.assertEqual(out["model_version_before"], 8)
        [(user, action, detail, extra)] = self.logs()
        self.assertEqual((user, action, detail),
                         ("op1", "feedback_candidate_queued",
                          "detection_id=1 type=false_negative"))
        self.assertEqual(extra, {"feedback_id": out["id"], "detection_id": 1,
                                 "source_event_id": "evt-1", "category": "cat",
                                 "learning_status": "candidate_queued",
                                 "detection_model_version": 7})
        self.assert_clean(extra)
        self.engine.submit_feedback.assert_not_called()
        # 旧数据缺版本 → None，不以 current_version 补
        self.feedback(det_id=2, event="evt-legacy")
        self.assertIsNone(self.logs()[-1][3]["detection_model_version"])
        # 无 source_event_id → None
        self.feedback(det_id=2, event=None)
        self.assertIsNone(self.logs()[-1][3]["source_event_id"])

    def test_idempotent_retry_adds_no_log(self):
        first = self.feedback()
        again = self.feedback()
        self.assertEqual(first["id"], again["id"])
        self.assertEqual(len(self.logs()), 1)

    def test_review_submit_links_feedback_id(self):
        out = routes_review.api_review_submit(3, routes_review.ReviewSubmitRequest(
            label=1, operator="rev", comment="评论不入审计"))
        queued, = self.logs("feedback_candidate_queued")
        submit, = self.logs("review_submit")
        self.assertEqual(queued[3]["feedback_id"], out["feedback_id"])
        self.assertEqual(submit[3]["feedback_id"], out["feedback_id"])
        self.assertEqual(submit[:3], ("rev", "review_submit",
            "detection_id=3 label=1 origin_decision=gray review_stage=submitted"))
        self.assertEqual(submit[3], {"feedback_id": out["feedback_id"], "detection_id": 3,
                                     "review_stage": "submitted", "origin_decision": "gray",
                                     "detection_model_version": 7})
        self.assertEqual(queued[3]["detection_model_version"], 7)
        self.assert_clean(submit[3])
        self.assertEqual(out["review_stage"], "submitted")
        # 重复提交 / 非法阶段：409，不新增日志
        n = len(self.logs())
        for det_id in (3, 4):
            with self.assertRaises(HTTPException) as ctx:
                routes_review.api_review_submit(det_id, routes_review.ReviewSubmitRequest(label=0))
            self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(len(self.logs()), n)

    def test_invalidate_then_refeedback(self):
        first = self.feedback(event="evt-a")
        inv = routes_feedback.api_invalidate_feedback(first["id"], operator="op2")
        self.assertEqual((inv["invalidated"], inv["was_consumed"]), (True, False))
        second = self.feedback(event="evt-b")
        self.assertNotEqual(first["id"], second["id"])
        queued = self.logs("feedback_candidate_queued")
        self.assertEqual([q[3]["feedback_id"] for q in queued], [first["id"], second["id"]])
        self.assertEqual([q[3]["source_event_id"] for q in queued], ["evt-a", "evt-b"])
        invalidate, = self.logs("feedback_invalidate")
        self.assertEqual(invalidate[:3], ("op2", "feedback_invalidate",
                                          f"feedback_id={first['id']}"))
        self.assertEqual(invalidate[3], {"feedback_id": first["id"], "detection_id": 1,
                                         "source_event_id": "evt-a", "invalidated": True,
                                         "was_consumed": False})
        # 重复作废：返回旧契约，不新增日志
        n = len(self.logs())
        again = routes_feedback.api_invalidate_feedback(first["id"])
        self.assertEqual(again["note"], "该反馈已是作废状态")
        self.assertEqual(len(self.logs()), n)
        with self.assertRaises(HTTPException) as ctx:
            routes_feedback.api_invalidate_feedback(999)
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(len(self.logs()), n)


if __name__ == "__main__":
    unittest.main(verbosity=2)
