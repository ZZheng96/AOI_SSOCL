"""AC-014 单帧解释：逐框区域证据 / 回退口径 / explain 路由 / UI 文本（TEMP SQLite，不碰生产）。"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from algo.decision.hierarchical import box_evidence
from backend.db import database
from backend.db.models import Base, Detection


def _maps():
    a = np.zeros((32, 32), np.float32)
    a[4:10, 4:10] = 1.0          # slot a 热点在左上
    b = np.zeros((32, 32), np.float32)
    b[20:28, 20:28] = 1.0        # slot b 热点在右下
    return {"a": a, "b": b, "flat": np.zeros((32, 32), np.float32)}


class BoxEvidenceTest(unittest.TestCase):
    def test_region_dominant_and_image_only(self):
        print("[1/5] box_evidence 区域主导/平坦槽位/无正向证据", flush=True)
        boxes = [{"bbox": [4, 4, 10, 10]}, {"bbox": [20, 20, 28, 28]}, {"bbox": [12, 0, 18, 3]}]
        box_evidence(boxes, _maps(), weights={"a": 1.0, "b": 1.0}, image_slots=["trad"])
        self.assertEqual(boxes[0]["dominant_slot"], "a")
        self.assertEqual(boxes[1]["dominant_slot"], "b")
        ev = boxes[0]["evidence"]
        self.assertEqual(ev["level"], "region")
        self.assertEqual(ev["image_only_slots"], ["flat", "trad"])
        self.assertNotIn("flat", [s["slot"] for s in ev["slots"]])
        self.assertIn("非缺陷概率", ev["semantics"])
        # 空白区域：所有对比为负 → 无正向区域证据
        self.assertEqual(boxes[2]["dominant_slot"], "")
        self.assertEqual(boxes[2]["evidence"]["level"], "none")

    def test_weight_changes_dominant(self):
        boxes = [{"bbox": [0, 0, 32, 32]}]
        m = {"a": _maps()["a"], "b": _maps()["b"]}
        box_evidence(boxes, m, weights={"a": 0.0, "b": 1.0})
        # 整图框对比均为 0 → none；权重为 0 不得成为主导
        self.assertNotEqual(boxes[0]["dominant_slot"], "a")


class ToResultFallbackTest(unittest.TestCase):
    def test_image_fallback_keeps_source(self):
        print("[2/5] _to_result 无区域证据回退整图口径", flush=True)
        from backend.engine import DetectionEngine
        pipe = SimpleNamespace(weights={"a": 1.0, "b": 0.5}, decider=None)
        rec = {"slot_scores": {"a": 0.9, "b": 0.2}, "decision": "anomaly", "fused": 0.9,
               "boxes": [{"bbox": [1, 2, 3, 4], "area": 4, "source": "local_branch"},
                         {"bbox": [5, 6, 7, 8], "area": 4, "dominant_slot": "b",
                          "evidence": {"level": "region", "slots": []}}]}
        r = DetectionEngine._to_result(DetectionEngine.__new__(DetectionEngine), pipe, rec, 1.0)
        b0, b1 = r.defect_boxes
        self.assertEqual(b0["dominant_slot"], "a")
        self.assertEqual(b0["evidence"]["level"], "image_fallback")
        self.assertEqual(b0["source"], "local_branch")
        self.assertEqual(b1["dominant_slot"], "b")
        self.assertEqual(b1["evidence"]["level"], "region")


class ExplainTradTest(unittest.TestCase):
    def test_not_applicable_without_trad(self):
        print("[3/5] explain_trad 无 trad 槽位", flush=True)
        from algo.eval.offline import Pipeline
        p = Pipeline.__new__(Pipeline)
        p.slots = {}
        self.assertEqual(p.explain_trad(rgb=np.zeros((8, 8, 3), np.uint8))["status"], "not_applicable")


class ExplainRouteTest(unittest.TestCase):
    def test_route(self):
        print("[4/5] /detections/{id}/explain 路由（TEMP SQLite）", flush=True)
        from backend.api import routes_detect
        nt = {"model_version": 3, "decision": "anomaly", "threshold": .9, "gray_threshold": .8,
              "raw": {"a": 1.5}, "cal": {"a": .97}, "weights": {"a": 1.0},
              "router_w": {"a": 1.0}, "triggered_slot": "a",
              "decision_trace": ["L1:fused>=tau_high", "anomaly"],
              "box_evidence": [{"bbox": [1, 2, 3, 4], "dominant_slot": "a",
                                "evidence": {"level": "region", "slots": [], "image_only_slots": []}}]}
        with tempfile.TemporaryDirectory() as tmp:
            engine = create_engine(f"sqlite:///{Path(tmp) / 'ex.db'}", connect_args={"check_same_thread": False})
            Base.metadata.create_all(engine)
            factory = sessionmaker(bind=engine, expire_on_commit=False)
            with patch.object(database, "_engine", engine), patch.object(database, "_SessionLocal", factory):
                with database.session_scope() as s:
                    s.add(Detection(image_path="x.png", category="test", final_score=.95,
                                    is_anomaly=True, latency_ms=1, n_tiles=nt))
                app = FastAPI()
                app.include_router(routes_detect.router, prefix="/api")
                client = TestClient(app)
                self.assertEqual(client.get("/api/detections/99/explain").status_code, 404)
                fake = Mock()
                fake.explain_image.return_value = {"model_version": 4, "trad": {"status": "available"}}
                with patch("backend.engine.get_engine", return_value=fake):
                    d = client.get("/api/detections/1/explain").json()
                fake.explain_image.assert_called_once_with("test", "x.png")
                for k in ("raw_scores", "cal_scores", "weights", "actual_router", "decision_trace",
                          "box_evidence", "score_semantics", "trad"):
                    self.assertIn(k, d)
                self.assertAlmostEqual(d["score"], .95)
                self.assertEqual(d["cal_scores"], {"a": .97})
                self.assertEqual(d["box_evidence"][0]["dominant_slot"], "a")
                self.assertIn("version_note", d["trad"])
                self.assertNotIn("contribution", d)
                fake.explain_image.side_effect = RuntimeError("engine not ready")
                with patch("backend.engine.get_engine", return_value=fake):
                    d = client.get("/api/detections/1/explain").json()
                self.assertEqual(d["trad"]["status"], "unavailable")
                self.assertEqual(d["decision_trace"], nt["decision_trace"])
            engine.dispose()


class ExplainUiTest(unittest.TestCase):
    def test_format_and_dialog(self):
        print("[5/5] UI 文本 + ExplainDialog offscreen", flush=True)
        from PySide6.QtWidgets import QApplication
        from ui.pages.monitor_page import ExplainDialog, format_box_evidence, format_explain
        QApplication.instance() or QApplication([])
        bev = [{"bbox": [1, 2, 3, 4], "dominant_slot": "a", "source": "local_branch",
                "evidence": {"level": "region", "image_only_slots": ["trad"],
                             "semantics": "框内相对热图强度（逐槽归一化），非缺陷概率",
                             "slots": [{"slot": "a", "region_mean": .8, "region_peak": 1.0,
                                        "image_mean": .1, "contrast": .7, "weighted": .7}]}}]
        text, tip = format_box_evidence(bev, 1)
        self.assertIn("框内区域证据", text)
        self.assertIn("局部分支", text)
        self.assertIn("非缺陷概率", tip)
        self.assertEqual(format_box_evidence([], 2), ("2 个", ""))
        data = {"detection_id": 1, "decision": "anomaly", "model_version": 3, "score": .95,
                "score_semantics": "融合异常分（校准后加权），非缺陷概率",
                "raw_scores": {"a": 1.5}, "cal_scores": {"a": .97}, "weights": {"a": 1.0},
                "decision_trace": ["L1", "anomaly"], "box_evidence": bev,
                "trad": {"status": "available", "score": 2.0, "score_semantics": "距离",
                         "scoring": "grouped", "version_note": "版本不同",
                         "group_breakdown": [{"name": "glcm", "enabled": True, "distance": 3.0,
                                              "normal_p5": .1, "normal_p95": 1.0, "out_of_range": True}],
                         "top_dims": [{"dim": 7, "group": "glcm", "value": 5.0, "normal_p5": 0.0,
                                       "normal_p95": 1.0, "z": 4.0}],
                         "neighbors": [{"index": 0, "distance": 1.2, "path": "n.png"}]}}
        s = format_explain(data)
        for kw in ("【实际槽位】", "非缺陷概率", "L1 → anomaly", "【逐框证据】", "glcm", "超出",
                   "dim7", "参考近邻", "版本不同"):
            self.assertIn(kw, s)
        self.assertNotIn("增益", s.replace("非增益", ""))
        s2 = format_explain({**data, "trad": {"status": "not_applicable", "reason": "无 trad"}})
        self.assertIn("not_applicable", s2)
        dlg = ExplainDialog(data)
        self.assertEqual(dlg.windowTitle(), "图像解释")
        dlg.deleteLater()


if __name__ == "__main__":
    unittest.main()
