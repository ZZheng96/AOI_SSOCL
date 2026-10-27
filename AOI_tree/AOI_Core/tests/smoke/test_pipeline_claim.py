"""SQLite 模拟产线 claim、回队及检测事务边界。"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.db import database
from backend.db.models import Base, DataSource, Dataset, Detection, Image, WorkOrder, WorkOrderSource
from backend.pipeline.pipeline_service import PipelineService


class PipelineClaimTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.engine = create_engine(f"sqlite:///{Path(self.tmp.name) / 'test.db'}")
        self.addCleanup(self.engine.dispose)
        Base.metadata.create_all(self.engine)
        self.factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.old_engine, self.old_factory = database._engine, database._SessionLocal
        database._engine, database._SessionLocal = self.engine, self.factory
        self.addCleanup(self._restore)
        with database.session_scope() as s:
            source = DataSource(name="source")
            s.add(source)
            s.flush()
            ds = Dataset(name="batch", category="test", datasource_id=source.id)
            wo1 = WorkOrder(name="one", pipeline_status="running")
            wo2 = WorkOrder(name="two", pipeline_status="running")
            s.add_all([ds, wo1, wo2])
            s.flush()
            s.add_all([WorkOrderSource(workorder_id=w.id, datasource_id=source.id)
                       for w in (wo1, wo2)])
            img = Image(path="fake.jpg", category="test", dataset_id=ds.id)
            s.add(img)
            s.flush()
            self.image_id, self.wo1, self.wo2 = img.id, wo1.id, wo2.id
        self.pipeline = PipelineService()
        self.fake_engine = patch("backend.engine.get_engine", return_value=FakeEngine())
        self.fake_service = patch("backend.pipeline.service.get_detection_service",
                                  return_value=FakeDetectionService())
        self.archive_patch = patch("backend.pipeline.service.DetectionService._auto_archive")
        for mock in (self.fake_engine, self.fake_service, self.archive_patch):
            mock.start()
            self.addCleanup(mock.stop)

    def _restore(self):
        database._engine, database._SessionLocal = self.old_engine, self.old_factory

    def test_multi_workorder_and_no_duplicate(self):
        self.pipeline._consume_one()
        self.pipeline._consume_one()
        self.pipeline._consume_one()
        with database.session_scope() as s:
            rows = s.query(Detection).order_by(Detection.id).all()
            self.assertEqual({r.workorder_id for r in rows}, {self.wo1, self.wo2})
            self.assertEqual(len(rows), 2)
            self.assertEqual(len({r.claim_token for r in rows}), 2)
            self.assertEqual(s.get(Image, self.image_id).processing_status, "succeeded")

    def test_requeue_generation_and_failed_is_not_retried(self):
        self.pipeline._consume_one()
        with database.session_scope() as s:
            wo = s.get(WorkOrder, self.wo1)
            wo.queue_json = {"requeued_ids": {"batch": [self.image_id]},
                             "requeued_at": {"batch": "2030-01-01T00:00:00"}}
            s.get(WorkOrder, self.wo2).pipeline_status = "paused"
        self.pipeline._consume_one()
        self.pipeline._consume_one()
        with database.session_scope() as s:
            self.assertEqual(s.query(Detection).filter_by(workorder_id=self.wo1).count(), 2)
            self.assertEqual(s.query(Detection).filter_by(
                claim_requeued_at="2030-01-01T00:00:00").count(), 1)
            self.assertFalse(s.get(WorkOrder, self.wo1).queue_json["requeued_ids"])
            s.get(Image, self.image_id).processing_status = "failed"
        self.pipeline._consume_one()
        with database.session_scope() as s:
            self.assertEqual(s.query(Detection).count(), 2)

    def test_crash_claim_requires_manual_recovery_and_stale_persist_rejected(self):
        from fastapi import HTTPException
        from backend.api.routes_wo_stats import ClaimRecoveryRequest, api_recover_pipeline_claim
        with database.session_scope() as s:
            img = s.get(Image, self.image_id)
            img.processing_status = "claiming"
            img.claim_token = "old-token"
            img.claim_workorder_id = self.wo1
        self.pipeline._consume_one()
        with database.session_scope() as s:
            self.assertEqual(s.query(Detection).count(), 0)
        with self.assertRaises(HTTPException):
            api_recover_pipeline_claim(self.wo1, ClaimRecoveryRequest(
                image_id=self.image_id, claim_token="old-token"))
        with patch("backend.api.routes_wo_stats.log_action"):
            api_recover_pipeline_claim(self.wo1, ClaimRecoveryRequest(
                image_id=self.image_id, claim_token="old-token", worker_stopped=True))
        with self.assertRaises(ValueError):
            from backend.pipeline.service import DetectionService
            DetectionService._persist(Mock(), {"category": "test"}, self.image_id,
                                      "pipeline", self.wo1, "old-token")
        self.pipeline._consume_one()
        with database.session_scope() as s:
            self.assertEqual(s.query(Detection).count(), 1)
            self.assertEqual(s.get(Image, self.image_id).processing_status, "succeeded")

    def test_lifecycle_revokes_inflight_and_rejects_late_stream(self):
        from fastapi import HTTPException
        from backend.api.routes_wo_order import api_archive_workorder, api_unarchive_workorder
        from backend.api.routes_wo_stats import api_pipeline_control
        from backend.pipeline.service import DetectionService
        with database.session_scope() as s:
            img = s.get(Image, self.image_id)
            img.processing_status = "claiming"
            img.claim_token = "late"
            img.claim_workorder_id = self.wo1
            s.get(WorkOrder, self.wo1).pipeline_status = "auto_paused"
        with patch("backend.api.routes_wo_order.log_action"):
            api_archive_workorder(self.wo1)
        with database.session_scope() as s:
            self.assertEqual(s.get(Image, self.image_id).processing_status, "pending")
            self.assertIsNone(s.get(Image, self.image_id).claim_token)
            self.assertEqual(s.get(WorkOrder, self.wo1).pipeline_status, "stopped")
        with self.assertRaises(HTTPException):
            api_pipeline_control(self.wo1, "resume")
        with self.assertRaises(ValueError):
            DetectionService._persist(Mock(), {}, None, "stream", self.wo1)
        with patch("backend.api.routes_wo_order.log_action"):
            api_unarchive_workorder(self.wo1)
        with self.assertRaises(ValueError):
            DetectionService._persist(Mock(), {}, self.image_id, "pipeline", self.wo1, "late")

    def test_queue_summary_and_consumer_share_detect_group(self):
        from backend.api._wo_shared import _pipeline_summary
        from backend.api.routes_wo_stats import api_workorder_queue
        with database.session_scope() as s:
            img = s.get(Image, self.image_id)
            img.split, img.label = "train", "normal"
            template = Image(path="template.jpg", category="test", split="template",
                             dataset_id=img.dataset_id)
            test = Image(path="test.jpg", category="test", split="test",
                         dataset_id=img.dataset_id)
            s.add_all([template, test])
            s.flush()
            test_id = test.id
            s.get(WorkOrder, self.wo2).pipeline_status = "paused"
        queue = api_workorder_queue(self.wo1)
        self.assertEqual(sum(b["n_images"] for b in queue["items"]), 1)
        self.assertEqual(sum(b["n_pending"] for b in queue["items"]), 1)
        self.pipeline._consume_one()
        with database.session_scope() as s:
            self.assertEqual([d.image_id for d in s.query(Detection)], [test_id])
            self.assertEqual(_pipeline_summary(s, s.get(WorkOrder, self.wo1))["n_pending"], 0)
            self.assertEqual(_pipeline_summary(s, s.get(WorkOrder, self.wo2))["n_pending"], 1)

    def test_review_counts_match_queue_stage_and_open_alert(self):
        from backend.db.models import Feedback
        from backend.api._wo_shared import _n_pending_review
        from backend.api.routes_review import api_review_queue, api_review_stats
        with database.session_scope() as s:
            s.get(WorkOrder, self.wo1).review_enabled = True
            rows = [Detection(target_type="pipeline", category="test", image_path=f"review-{stage}.jpg",
                              final_score=0.0, is_anomaly=False, latency_ms=0.0,
                              workorder_id=self.wo1,
                              n_tiles={"decision": "normal", "review_stage": stage})
                    for stage in ("queued", "submitted", "invalid")]
            s.add_all(rows)
            s.flush()
            s.add(Feedback(detection_id=rows[0].id, feedback_type="open_alert",
                           operator_label="ignored"))
        queue, stats = api_review_queue(), api_review_stats()
        self.assertEqual(queue["total"], 1)
        self.assertEqual(stats["pending_total"], queue["total"])
        with database.session_scope() as s:
            self.assertEqual(_n_pending_review(s, self.wo1), 1)


class FakeEngine:
    def current_version(self, category):
        return 1

    def get_pipeline(self, category):
        return object()


class FakeDetectionService:
    def detect_image(self, path, category, *, image_id, workorder_id, claim_token, **kwargs):
        from backend.pipeline.service import DetectionService
        det = {"category": category, "image_path": path, "final_score": 0.2,
               "is_anomaly": False, "latency_ms": 1.0}
        return DetectionService._persist(Mock(), det, image_id, "pipeline",
                                         workorder_id, claim_token)


if __name__ == "__main__":
    unittest.main()
