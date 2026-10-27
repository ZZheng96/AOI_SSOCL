"""TEMP SQLite + offscreen契约；engine/训练为替身，不证明生产归因或GPU学习。"""
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.db import database
from backend.db.models import Base, Detection, Feedback, Image
from backend.self_learning import service
from backend.api import routes_feedback
from backend.api.schemas import FeedbackCreateRequest
from ui.widgets.feedback_dialog import FeedbackDialog, FeedbackEvidenceDialog, feedback_evidence_text


class FeedbackEvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="aoi_evidence_")
        self.addCleanup(self.tmp.cleanup)
        self.db = create_engine("sqlite:///" + os.path.join(self.tmp.name, "evidence.sqlite"))
        self.addCleanup(self.db.dispose)
        self.sessions = sessionmaker(bind=self.db, expire_on_commit=False)
        Base.metadata.create_all(self.db)
        with self.sessions.begin() as s:
            s.add(Image(path="fixture.png", category="cat", split="feedback", label="anomaly"))
            s.add(Detection(id=1, image_path="fixture.png", category="cat", target_type="review",
                            final_score=.2, is_anomaly=False, latency_ms=1))
        self.engine = SimpleNamespace(current_version=Mock(return_value=1),
            predict_image_path=Mock(return_value=SimpleNamespace(score=.2, decision="normal", slot_scores={})))
        self.update = {"action": "difficult_queued", "drift": .1, "boost": .2, "defect_ratio": .3,
            "attribution": {"mode": "fn", "verdict": "undertrained_fusion", "next": "真实算法下一阶梯",
                "series": [{"step": "signal_presence", "top_slot": "sem", "top_cal": .6, "ok": True}]}}
        def submit(*args, **kwargs):
            kwargs["on_result"](self.update, None)
            return self.update
        self.engine.submit_feedback = Mock(side_effect=submit)
        for p in (patch.object(database, "get_session", self.sessions),
                  patch.object(service, "get_engine", return_value=self.engine),
                  patch.object(service, "notify_event"), patch.object(service, "log_action")):
            p.start()
            self.addCleanup(p.stop)

    def request(self, source_event_id="evt"):
        return routes_feedback.api_create_feedback(FeedbackCreateRequest(
            detection_id=1, feedback_type="false_negative", operator_label=1,
            source_event_id=source_event_id))

    def legacy_request(self, source_event_id="evt"):
        return service.submit_feedback_record(
            1, "false_negative", 1, source_event_id=source_event_id,
            candidate_only=False, allow_legacy_online=True)

    def test_production_feedback_is_candidate_only(self):
        live = self.request()
        self.assertEqual(live["learning_status"], "candidate_queued")
        self.assertIsNone(live["engine_update"])
        self.engine.submit_feedback.assert_not_called()
        self.assertFalse(routes_feedback.api_feedback_details(live["id"])["consumed"])

    def test_submit_success_does_not_open_long_lived_evidence_dialog(self):
        dialog = FeedbackDialog.__new__(FeedbackDialog)
        dialog.submitted = Mock()
        dialog._set_status = Mock()
        dialog.accept = Mock()
        with patch("ui.widgets.feedback_dialog.FeedbackEvidenceDialog") as evidence:
            FeedbackDialog._on_done(dialog, {"learning_status": "candidate_queued"})
        evidence.assert_not_called()
        dialog.submitted.emit.assert_called_once()
        dialog.accept.assert_called_once()

    def test_live_history_duplicate_and_offscreen(self):
        live = self.legacy_request()
        history = routes_feedback.api_feedback_details(live["feedback_id"])
        self.assertEqual(live["learning_trace"]["engine_update"], history["details"]["engine_update"])
        self.assertEqual(history["details"]["attribution"], self.update["attribution"])
        self.assertEqual(self.legacy_request()["learning_trace"], live["learning_trace"])
        self.assertEqual(self.engine.submit_feedback.call_count, 1)
        calls = self.engine.predict_image_path.call_count
        routes_feedback.api_feedback_details(live["feedback_id"])
        self.assertEqual(calls, self.engine.predict_image_path.call_count)
        self.assertEqual(routes_feedback.api_list_feedback()["items"][0]["details"], history["details"])
        for data in (live, history):
            dialog = FeedbackEvidenceDialog(data)
            text = dialog.evidence.toPlainText()
            self.assertIn("signal_presence", text)
            self.assertIn("真实算法下一阶梯", text)
            self.assertNotIn("confidence：未知", text)
            self.assertNotIn("source_slot：未知", text)
            self.assertIn("独立 evidence：\n不适用（未记录）", text)
            dialog.close()

    def test_real_handler_engine_attribution_and_optional_evidence(self):
        from backend.engine import DetectionEngine
        from algo.ssocl.feedback import FeedbackHandler
        from algo.decide import Decider
        print("[TEMP attribution] real handler -> engine -> finite JSON", flush=True)
        engine = DetectionEngine(self.tmp.name, "configs/engine_fast.yaml")
        pipe = SimpleNamespace(slots={"trad": object()}, weights={"trad": 1.0},
            cfg={}, decider=Decider(.9, .7), _ablate_intercept=True, head_mgr=None,
            _get_item=lambda *a: {"feats": None},
            _record=lambda item: {"slot_scores": {"trad": .6}, "raw_scores": {"trad": .8},
                                  "fused": .2}, _ablate_reflow=False)
        handler = FeedbackHandler(pipe, {"head_ft": {"enabled": False},
                                        "disc_ft": {"enabled": False}})
        pipe.handler = handler
        pipe.feedback = handler.feedback
        engine._cache_put("cat", pipe, 1)
        with patch.object(handler, "_add_defect_sample"), patch.object(handler, "_maybe_finetune_router"):
            out = engine.submit_feedback("cat", "fixture.png", verdict="wrong", label=1, event_id="real")
        self.assertEqual(out["attribution"], handler.feedback_log[-1]["attribution"])
        self.assertEqual(out["attribution"]["verdict"], "undertrained_fusion")
        self.assertEqual(out["attribution"]["series"][1]["fused"], .2)
        self.assertNotIn("evidence", out)
        self.assertNotIn("confidence", out["attribution"])
        self.assertEqual(service.feedback_json(out)["attribution"], out["attribution"])
        # 独立evidence当前handler不产生；注入仅验证存在时原样透传。
        evidence = {"score": float("inf"), "valid_score": .73}
        with patch.object(pipe, "feedback", return_value={"evidence": evidence}):
            optional = engine.submit_feedback("cat", "fixture.png", verdict="none", event_id="optional")
        self.assertIs(optional["evidence"], evidence)
        self.assertEqual(service.feedback_json(optional)["evidence"], {"score": None, "valid_score": .73})
        self.assertNotIn("attribution", optional)

    def test_callback_before_pending_return_and_duplicate(self):
        def submit(*args, **kwargs):
            kwargs["on_result"](self.update, None)
            kwargs["on_result"](None, RuntimeError("late duplicate"))
            return {"async": True}
        self.engine.submit_feedback.side_effect = submit
        live = self.legacy_request()
        self.assertEqual(live["learning_status"], "applied")
        self.assertEqual(live["learning_trace"]["engine_update"]["attribution"], self.update["attribution"])
        self.assertEqual(self.legacy_request()["learning_trace"], live["learning_trace"])
        self.assertEqual(self.engine.submit_feedback.call_count, 1)

    def test_late_result_success_and_failure(self):
        import threading
        for failed in (False, True):
            with self.subTest(failed=failed):
                callbacks = []
                def submit(*args, **kwargs):
                    callbacks.append(kwargs["on_result"])
                    return {"async": True}
                self.engine.submit_feedback.side_effect = submit
                live = service.submit_feedback_record(
                    1, "false_negative", 1, source_event_id=f"late-{failed}",
                    candidate_only=False, allow_legacy_online=True)
                self.assertEqual(live["learning_status"], "pending")
                calls = self.engine.submit_feedback.call_count
                duplicate = service.submit_feedback_record(
                    1, "false_negative", 1, source_event_id=f"late-{failed}",
                    candidate_only=False, allow_legacy_online=True)
                self.assertEqual(duplicate["learning_status"], "pending")
                self.assertEqual(calls, self.engine.submit_feedback.call_count)
                errors = []
                def finish():
                    try:
                        callbacks[0](None if failed else self.update,
                                     RuntimeError("background failed") if failed else None)
                        callbacks[0](self.update, None)
                    except Exception as e:
                        errors.append(e)
                worker = threading.Thread(target=finish)
                worker.start()
                worker.join(3)
                self.assertFalse(worker.is_alive())
                self.assertEqual(errors, [])
                data = routes_feedback.api_feedback_details(live["feedback_id"])
                self.assertEqual(data["details"]["learning_status"], "failed" if failed else "applied")
                self.assertEqual(data["consumed"], not failed)
                self.assertEqual(data["details"]["attribution"], None if failed else self.update["attribution"])
                json.dumps(data, allow_nan=False)

    def test_real_engine_timeout_final_writeback(self):
        import threading
        from backend.engine import DetectionEngine
        for failed in (False, True):
            with self.subTest(failed=failed):
                entered, release, finished = threading.Event(), threading.Event(), threading.Event()
                def learn(*args, **kwargs):
                    entered.set()
                    if not release.wait(3):
                        raise RuntimeError("test release timeout")
                    if failed:
                        raise RuntimeError("background failed")
                    return {**self.update, "evidence": {"score": float("nan"), "valid": .5}}
                pipe = SimpleNamespace(feedback=Mock(side_effect=learn), handler=None)
                engine = DetectionEngine.__new__(DetectionEngine)
                engine.storage_dir = self.tmp.name
                engine.snap_root = os.path.join(self.tmp.name, "snapshots")
                engine._ensure_loaded = Mock()
                engine._pipe_entry = Mock(return_value=(pipe, 1))
                engine._pipe_lock = Mock(return_value=threading.RLock())
                engine.current_version = self.engine.current_version
                engine.predict_image_path = self.engine.predict_image_path
                engine.feedback_timeout_ms = 1
                original = engine.submit_feedback
                def submit(*args, **kwargs):
                    callback = kwargs["on_result"]
                    def result(update, error):
                        try:
                            callback(update, error)
                        finally:
                            finished.set()
                    kwargs["on_result"] = result
                    return original(*args, **kwargs)
                engine.submit_feedback = Mock(side_effect=submit)
                with patch.object(service, "get_engine", return_value=engine):
                    live = service.submit_feedback_record(1, "false_negative", 1,
                        source_event_id=f"engine-{failed}", candidate_only=False,
                        allow_legacy_online=True)
                    try:
                        self.assertTrue(entered.wait(2))
                        self.assertEqual(live["learning_status"], "pending")
                        duplicate = service.submit_feedback_record(1, "false_negative", 1,
                            source_event_id=f"engine-{failed}", candidate_only=False,
                            allow_legacy_online=True)
                        self.assertEqual(duplicate["learning_status"], "pending")
                    finally:
                        release.set()
                    self.assertTrue(finished.wait(3))
                    data = routes_feedback.api_feedback_details(live["feedback_id"])
                    self.assertEqual(data["details"]["learning_status"], "failed" if failed else "applied")
                    self.assertEqual(data["consumed"], not failed)
                    self.assertEqual(engine.submit_feedback.call_count, 1)
                    self.assertEqual(pipe.feedback.call_count, 1)
                    if not failed:
                        self.assertEqual(data["details"]["attribution"], self.update["attribution"])
                        self.assertEqual(data["details"]["evidence"], {"score": None, "valid": .5})
                    else:
                        self.assertIsNone(data["details"]["attribution"])
                    json.dumps(data, allow_nan=False)

    def test_legacy_unknown_no_inference(self):
        with self.sessions.begin() as s:
            s.add(Feedback(id=9, detection_id=1, feedback_type="confirmed", operator_label=0, consumed=True))
        data = routes_feedback.api_feedback_details(9)
        self.engine.predict_image_path.assert_not_called()
        self.assertEqual(data["details"]["attribution_availability"], "unavailable")
        text = feedback_evidence_text(data)
        self.assertIn("归因证据：不适用或未记录", text)
        self.assertIn("归因详情：暂无真实归因记录", text)
        self.assertNotIn("归因 verdict：未知", text)

    def test_finite_json_preserves_fields(self):
        self.update["evidence"] = {"nested": [float("nan"), float("inf"), {"score": .5}]}
        data = self.legacy_request("finite")
        json.dumps(data, allow_nan=False)
        stored = routes_feedback.api_feedback_details(data["feedback_id"])
        self.assertEqual(stored["details"]["evidence"]["nested"], [None, None, {"score": .5}])
        for key in ("action", "drift", "boost", "defect_ratio"):
            self.assertEqual(stored["details"]["engine_update"][key], self.update[key])

    def test_failure_and_async_do_not_claim_complete(self):
        self.engine.submit_feedback.side_effect = RuntimeError("learn failed")
        failed = self.legacy_request("failed")
        self.assertEqual(failed["learning_status"], "failed")
        self.assertNotIn("已学习", FeedbackDialog._success_text(failed))
        self.assertIsNone(routes_feedback.api_feedback_details(failed["feedback_id"])["details"]["attribution"])
        self.engine.submit_feedback.side_effect = None
        self.engine.submit_feedback.return_value = {"async": True}
        pending = self.legacy_request("pending")
        self.assertEqual(pending["learning_status"], "pending")
        self.assertIn("不代表学习完成", FeedbackDialog._success_text(pending))


if __name__ == "__main__":
    unittest.main(verbosity=2)
