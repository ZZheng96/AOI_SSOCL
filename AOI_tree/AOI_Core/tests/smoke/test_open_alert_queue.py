"""TEMP SQLite / API / offscreen 最小闭环，不触碰生产资产。"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.db import database
from backend.db.models import Base, Detection, Feedback
from backend.api.open_alerts import register, new_item


class OpenQueueTest(unittest.TestCase):
    def test_temp_api_and_offscreen(self):
        print("[1/3] TEMP SQLite + API", flush=True)
        with tempfile.TemporaryDirectory() as tmp:
            engine = create_engine(f"sqlite:///{Path(tmp) / 'queue.db'}", connect_args={"check_same_thread": False})
            Base.metadata.create_all(engine)
            factory = sessionmaker(bind=engine, expire_on_commit=False)
            from algo.feature_revision import FeatureSetRevision
            store = FeatureSetRevision(Path(tmp) / "rev" / "registry.json")
            with patch.object(database, "_engine", engine), patch.object(database, "_SessionLocal", factory), \
                    patch("backend.api.open_alerts._revision_store", return_value=store):
                with database.session_scope() as s:
                    for nt in ({"open_alert": True, "model_version": 7,
                                "open_alert_item": new_item(3.2, 2.5)}, {"open_alert": True}, {}):
                        s.add(Detection(image_path="missing.png", category="test", final_score=.1,
                                        is_anomaly=False, latency_ms=1, n_tiles=nt))
                app = FastAPI()
                from fastapi import APIRouter
                router = APIRouter()
                register(router)
                app.include_router(router, prefix="/api")
                client = TestClient(app)
                listing = client.get("/api/review/open_alerts").json()
                self.assertEqual(listing["total"], 2)
                self.assertEqual(client.get("/api/review/open_alerts/2").json()["evidence_status"], "legacy_unknown")
                self.assertIsNone(client.get("/api/review/open_alerts/2").json()["e_open_score"])
                self.assertEqual(client.get("/api/review/open_alerts/3").status_code, 404)
                self.assertEqual(client.get("/api/review/open_alerts?page=0").status_code, 422)
                self.assertEqual(client.get("/api/review/open_alerts?status=queued").json()["total"], 2)
                self.assertEqual(client.get("/api/review/open_alerts?page_size=1&page=2").json()["items"][0]["detection_id"], 1)
                self.assertEqual(client.get("/api/review/open_alerts?category=absent").json()["total"], 0)
                self.assertEqual(client.get("/api/review/open_alerts?workorder_id=99").json()["total"], 0)
                self.assertEqual(client.get("/api/review/open_alerts?status=invalid").status_code, 422)
                print("[2/3] 状态、幂等、冲突与反馈事件", flush=True)
                payload = {"status": "confirmed_defect", "operator": "tester", "reason": "人工确认",
                           "event_id": "event1", "expected_revision": 0}
                url = "/api/review/open_alerts/1/disposition"
                for field in ("operator", "reason", "event_id"):
                    self.assertEqual(client.post(url, json={**payload, field: " "}).status_code, 422)
                first = client.post(url, json=payload)
                self.assertEqual(first.status_code, 200, first.text)
                self.assertTrue(first.json()["item"]["is_defect"])
                self.assertTrue(client.post(url, json=payload).json()["idempotent"])
                self.assertEqual(client.post(url, json={**payload, "reason": "different"}).status_code, 409)
                self.assertEqual(client.post(url, json={**payload, "event_id": "other"}).status_code, 409)
                for revision, status in enumerate(("confirmed_normal", "template_candidate", "deferred", "ignored"), 1):
                    response = client.post(url, json={**payload, "status": status,
                        "event_id": f"event{revision+1}", "expected_revision": revision})
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertFalse(response.json()["item"]["is_defect"])
                self.assertEqual(client.get("/api/review/open_alerts?status=queued").json()["total"], 1)
                self.assertEqual(client.get("/api/review/open_alerts?status=ignored").json()["total"], 1)
                # AC-034 原因归类：矛盾组合 422；new_defect 只登记候选准入事件，不自动上线
                p2 = {**payload, "event_id": "c1", "expected_revision": 5}
                self.assertEqual(client.post(url, json={**p2, "status": "confirmed_normal",
                                                        "cause": "new_defect"}).status_code, 422)
                self.assertEqual(client.post(url, json={**p2, "cause": "drift"}).status_code, 422)
                self.assertEqual(client.post(url, json={**p2, "cause": "bogus"}).status_code, 422)
                r = client.post(url, json={**p2, "cause": "new_defect"})
                self.assertEqual(r.status_code, 200, r.text)
                adm = r.json()["item"]["admission"]
                self.assertEqual((adm["status"], adm["auto_online"]), ("candidate", False))
                self.assertIn("version", adm["required_checks"])
                self.assertTrue(client.post(url, json={**p2, "cause": "new_defect"}).json()["idempotent"])
                r = client.post(url, json={**p2, "status": "confirmed_normal", "cause": "drift",
                                           "event_id": "c2", "expected_revision": 6})
                self.assertIsNone(r.json()["item"]["admission"])
                with database.session_scope() as s:
                    tr = [f.learning_trace for f in s.query(Feedback).order_by(Feedback.id)][-2:]
                    self.assertEqual([t["open_cause"] for t in tr], ["new_defect", "drift"])
                    self.assertTrue(all(t["candidate_only"] for t in tr))
                with database.session_scope() as s:
                    self.assertEqual(s.query(Feedback).count(), 7)
                    self.assertTrue(all(not f.consumed for f in s.query(Feedback)))
                    self.assertEqual(s.get(Detection, 1).n_tiles["open_alert_item"]["revision"], 7)
                print("[3/3] Qt offscreen 展示 + 处置提交", flush=True)
                from PySide6.QtWidgets import QApplication, QDialog, QLineEdit
                from ui.widgets.open_alert_queue import OpenAlertQueue
                appqt = QApplication.instance() or QApplication([])
                fake = Mock()
                widget = OpenAlertQueue(fake, lambda: None)
                widget.fill(listing)
                self.assertEqual(widget.table.rowCount(), 2)
                self.assertIn("旧证据未知", widget.table.item(0, 5).text())
                def accept(dialog):
                    edits = dialog.findChildren(QLineEdit)
                    edits[-1].setText("offscreen 人工处置")
                    return QDialog.Accepted
                item = client.get("/api/review/open_alerts/1").json()
                item["image_path"] = ""
                fake.post.return_value = {}
                with patch("ui.widgets.open_alert_queue.run_async", side_effect=lambda owner, fn, callback: fn()), patch.object(QDialog, "exec", accept):
                    widget._detail(item)
                self.assertEqual(fake.post.call_args.kwargs["json"]["expected_revision"], 7)
                widget.close()
                appqt.processEvents()
            engine.dispose()

    def test_permissions_and_persisted_evidence(self):
        from fastapi import APIRouter
        from backend.core.security import ApiKeyMiddleware
        from backend.engine import DetectionEngine
        from backend.pipeline.service import DetectionService
        from types import SimpleNamespace
        print("[权限/血缘] TEMP 检测落库 + 真实鉴权中间件", flush=True)
        with tempfile.TemporaryDirectory() as tmp:
            engine = create_engine(f"sqlite:///{Path(tmp) / 'evidence.db'}", connect_args={"check_same_thread": False})
            Base.metadata.create_all(engine)
            factory = sessionmaker(bind=engine, expire_on_commit=False)
            settings = Mock()
            settings.section.return_value = {"keys": {"test-op": "operator", "test-eng": "engineer", "test-admin": "admin"}}
            try:
                with patch.object(database, "_engine", engine), patch.object(database, "_SessionLocal", factory), \
                        patch("backend.core.security.get_settings", return_value=settings), \
                        patch("backend.core.security.get_configured_api_key", return_value=""), \
                        patch("backend.pipeline.service.get_engine", side_effect=AssertionError("禁止重读当前模型")), \
                        patch("backend.pipeline.service.notify_event"):
                    pipe = SimpleNamespace(weights={}, decider=SimpleNamespace(tau_high=.9, tau_gray=.7), open_thresh=2.5)
                    result = DetectionEngine._to_result(DetectionEngine.__new__(DetectionEngine), pipe, {"fused": .1, "decision": "normal", "open_score": 3.2,
                        "open_alert": True}, 1.0, 7)
                    service = DetectionService.__new__(DetectionService)
                    service._record_alarm = Mock(return_value={"active": False})
                    import time
                    det = service._finish_detect(result, None, "evidence.png", "test", time.perf_counter(),
                                                 persist=False, with_heatmap=False)
                    detection_id = DetectionService._persist(Mock(), det, None, "image")
                    app = FastAPI()
                    app.add_middleware(ApiKeyMiddleware)
                    router = APIRouter()
                    register(router)
                    app.include_router(router, prefix="/api")
                    url = f"/api/review/open_alerts/{detection_id}"
                    payload = {"status": "confirmed_normal", "operator": "责任人甲", "reason": "人工复判正常",
                               "event_id": "audit1", "expected_revision": 0}
                    with TestClient(app) as client:
                        self.assertEqual(client.get(url).status_code, 401)
                        self.assertEqual(client.post(url + "/disposition", json=payload).status_code, 401)
                        self.assertEqual(client.get(url, headers={"X-API-Key": "wrong"}).status_code, 401)
                        op = {"X-API-Key": "test-op"}
                        eng = {"X-API-Key": "test-eng"}
                        self.assertEqual(client.get(url, headers=op).status_code, 200)
                        self.assertEqual(client.post(url + "/disposition", headers=op, json=payload).status_code, 403)
                        item = client.get(url, headers=eng).json()
                        self.assertEqual((item["model_version"], item["e_open_score"], item["e_open_threshold"]), (7, 3.2, 2.5))
                        self.assertEqual(item["image_path"], "evidence.png")
                        self.assertFalse(item["is_defect"])
                        first = client.post(url + "/disposition", headers=eng, json=payload)
                        self.assertEqual(first.status_code, 200, first.text)
                        self.assertTrue(client.post(url + "/disposition", headers={"X-API-Key": "test-admin"}, json=payload).json()["idempotent"])
                    with database.session_scope() as s:
                        saved = s.get(Detection, detection_id)
                        state = saved.n_tiles["open_alert_item"]
                        self.assertEqual(state["operator"], "责任人甲")
                        self.assertEqual(state["reason"], "人工复判正常")
                        self.assertTrue(state["disposed_at"])
                        self.assertEqual(len(state["events"]), 1)
                        self.assertFalse(saved.is_anomaly)
                        fb = s.query(Feedback).one()
                        self.assertEqual(state["feedback_id"], fb.id)
                        self.assertEqual(fb.model_version_before, "7")
                        self.assertFalse(fb.consumed)
                        self.assertEqual(fb.learning_trace["learning_status"], "not_requested")
                    engine.dispose()
                    with TestClient(app) as restarted:
                        self.assertEqual(restarted.get(url, headers=eng).json()["revision"], 1)
            finally:
                engine.dispose()


if __name__ == "__main__":
    unittest.main(verbosity=2)
