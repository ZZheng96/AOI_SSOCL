"""AC-063~065 / LOC-1~5：区域记录、渲染不覆盖引擎框、区域复核修订链、定位评估。"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from algo.decision import localization as L
from backend.db import database
from backend.db.models import Base, Detection


def _rec_with_boxes():
    params = L.postproc_params({"hierarchical": {"top_k": 2}})
    fused = np.zeros((100, 100), np.float32)
    fused[10:20, 10:20] = 0.9
    fused[50:60, 50:60] = 0.4
    fused[80:90, 80:90] = 0.2
    boxes = [{"bbox": [50, 50, 60, 60], "area": 100}, {"bbox": [10, 10, 20, 20], "area": 100},
             {"bbox": [80, 80, 90, 90], "area": 100}]
    regions = L.finalize_regions(boxes, fused, 0.3, (300, 290), params)
    summ = L.localization_summary(regions, params, (300, 290))
    return params, regions, summ


class LocalizationPureTest(unittest.TestCase):
    def test_postproc_version_and_finalize(self):
        print("[1/4] 后处理版本 + 区域记录", flush=True)
        a, b = L.postproc_params({}), L.postproc_params({"hierarchical": {}})
        self.assertEqual(a["version_id"], b["version_id"])
        self.assertNotEqual(a["version_id"], L.postproc_params({"hierarchical": {"min_area": 32}})["version_id"])
        params, regions, summ = _rec_with_boxes()
        self.assertEqual([r["region_id"] for r in regions], ["r0", "r1"])  # top_k=2，按峰值降序
        self.assertAlmostEqual(regions[0]["score_peak"], 0.9, places=5)
        r0 = regions[0]
        self.assertEqual(r0["bbox"], [33, 33, 67, 67])  # floor(10/0.3)=33, ceil(20/0.3)=67
        self.assertEqual((r0["source"], r0["status"], r0["confirmed"], r0["postproc"]),
                         ("engine_box", "candidate", False, params["version_id"]))
        self.assertEqual(r0["area_work"], 100)
        # 越界裁剪：图 290 宽
        clip = L.finalize_regions([{"bbox": [90, 90, 100, 100], "area": 1}], None, 0.3, (300, 290), params)
        self.assertLessEqual(clip[0]["bbox"][2], 290)
        self.assertLessEqual(clip[0]["bbox"][3], 300)
        self.assertIsNone(clip[0]["score_peak"])
        # 已带来源（local_branch）不被覆盖
        lb = L.finalize_regions([{"bbox": [1, 1, 5, 5], "area": 16, "source": "local_branch"}], None, 1.0, (10, 10), params)
        self.assertEqual(lb[0]["source"], "local_branch")
        self.assertEqual((summ["status"], summ["n_regions"], summ["coord_space"]["size"]), ("ok", 2, [290, 300]))
        self.assertEqual(L.localization_summary([], params, (10, 10))["status"], "no_region")

    def test_review_ops(self):
        print("[2/4] 区域复核纯函数", flush=True)
        _, regions, _ = _rec_with_boxes()
        rv = L.new_review(regions, "pp-x")
        self.assertEqual(L.trainable_region_boxes(rv), [])
        base = {"operator": "甲", "expected_revision": 0}
        rv1 = L.apply_op(rv, {**base, "op": "confirm", "region_id": "r0", "event_id": "e1"})
        rv2 = L.apply_op(rv1, {**base, "op": "reject", "region_id": "r1", "event_id": "e2"})
        rv3 = L.apply_op(rv2, {**base, "op": "add", "bbox": [1, 1, 9, 9], "event_id": "e3"}, size=[290, 300])
        rv4 = L.apply_op(rv3, {**base, "op": "revise", "region_id": "r1", "bbox": [0, 0, 5, 5], "event_id": "e4"})
        self.assertEqual(rv4["revision"], 4)
        self.assertEqual(len(rv4["events"]), 4)
        self.assertEqual(rv["revision"], 0)  # 纯函数不改入参
        r1 = next(r for r in rv4["regions"] if r["region_id"] == "r1")
        self.assertEqual((r1["source"], r1["status"], r1["revised_from"]["source"]), ("human_box", "confirmed", "engine_box"))
        h0 = next(r for r in rv4["regions"] if r["region_id"] == "h0")
        self.assertEqual((h0["source"], h0["operator"], h0["event_id"]), ("human_box", "甲", "e3"))
        self.assertEqual(len(L.trainable_region_boxes(rv4)), 3)
        self.assertEqual(L.trainable_region_boxes(rv2), [regions[0]["bbox"]])
        with self.assertRaises(ValueError):
            L.apply_op(rv, {**base, "op": "add", "bbox": [5, 5, 1, 1], "event_id": "x"})
        with self.assertRaises(ValueError):
            L.apply_op(rv, {**base, "op": "add", "bbox": [0, 0, 999, 9], "event_id": "x"}, size=[290, 300])
        with self.assertRaises(ValueError):
            L.apply_op(rv, {**base, "op": "delete", "event_id": "x"})
        with self.assertRaises(KeyError):
            L.apply_op(rv, {**base, "op": "confirm", "region_id": "nope", "event_id": "x"})

    def test_eval_and_select(self):
        print("[3/4] 定位评估 + dev 选择", flush=True)
        res = L.eval_localization([
            {"pred": [[0, 0, 10, 10], [50, 50, 60, 60]], "gt": [[0, 0, 10, 10]]},
            {"pred": [[30, 30, 40, 40]], "gt": [[0, 0, 10, 10]]},
            {"pred": [], "gt": []},
        ], postproc="pp-a")
        self.assertEqual((res["tp"], res["n_gt"], res["n_pred"]), (1, 2, 3))
        self.assertAlmostEqual(res["instance_recall"], 0.5)
        self.assertAlmostEqual(res["precision"], 1 / 3)
        self.assertAlmostEqual(res["mislocalization_rate"], 0.5)
        self.assertAlmostEqual(res["mean_iou"], 1.0)
        self.assertAlmostEqual(L.iou([0, 0, 10, 10], [5, 0, 15, 10]), 1 / 3)
        cands = [({"v": "a"}, {"instance_recall": 0.5, "precision": 0.2}),
                 ({"v": "b"}, {"instance_recall": 0.8, "precision": 0.1})]
        self.assertEqual(L.select_postproc(cands, "dev")["params"]["v"], "b")
        for split, metric in (("test", "instance_recall"), ("dev", "auroc")):
            with self.assertRaises(ValueError):
                L.select_postproc(cands, split, metric)


class LocalizationServiceApiTest(unittest.TestCase):
    def test_result_render_persist_and_api(self):
        from backend.core.security import ApiKeyMiddleware
        from backend.engine import DetectionEngine
        from backend.pipeline.service import DetectionService, render_heatmap_overlay
        from backend.api.region_review import register
        from backend.api.schemas import to_dict
        print("[4/4] 引擎透出 → 渲染 → 落库 → 区域复核 API", flush=True)
        params, regions, summ = _rec_with_boxes()
        fused = np.zeros((100, 100), np.float32)
        fused[10:20, 10:20] = 0.9
        pipe = SimpleNamespace(weights={}, decider=SimpleNamespace(tau_high=.9, tau_gray=.7), open_thresh=2.5)
        rec = {"fused": .95, "decision": "anomaly", "boxes": regions,
               "fused_heatmap": fused, "localization": summ}
        res = DetectionEngine._to_result(DetectionEngine.__new__(DetectionEngine), pipe, rec, 1.0, 3)
        b0 = res.defect_boxes[0]
        for k in ("source", "region_id", "score_peak", "postproc", "status"):
            self.assertIn(k, b0)
        self.assertFalse(b0["confirmed"])
        self.assertIs(res.fused_heatmap, fused)
        self.assertEqual(res.extra["localization"]["postproc"]["version_id"], params["version_id"])
        legacy = DetectionEngine._to_result(DetectionEngine.__new__(DetectionEngine), pipe, {"fused": .1}, 1.0, 3)
        self.assertEqual(legacy.extra["localization"]["status"], "legacy_unknown")

        heat, ov, hb = render_heatmap_overlay(np.zeros((300, 290, 3), np.uint8), fused,
                                              engine_boxes=[[33, 33, 67, 67]])
        self.assertEqual(ov.shape, (300, 290, 3))
        self.assertTrue(hb and len(hb[0]) == 4)
        self.assertEqual(ov[33, 40].tolist(), [0, 0, 255])  # 红色引擎框

        with tempfile.TemporaryDirectory() as tmp:
            engine = create_engine(f"sqlite:///{Path(tmp) / 'loc.db'}", connect_args={"check_same_thread": False})
            Base.metadata.create_all(engine)
            factory = sessionmaker(bind=engine, expire_on_commit=False)
            settings = Mock()
            settings.get.side_effect = lambda sec, key, default=None: default
            settings.storage.return_value = Path(tmp)
            sec = Mock()
            sec.section.return_value = {"keys": {"t-op": "operator", "t-eng": "engineer"}}
            try:
                with patch.object(database, "_engine", engine), patch.object(database, "_SessionLocal", factory), \
                        patch("backend.core.security.get_settings", return_value=sec), \
                        patch("backend.core.security.get_configured_api_key", return_value=""), \
                        patch("backend.pipeline.service.get_engine", side_effect=AssertionError("禁止重读模型")), \
                        patch("backend.pipeline.service.notify_event"):
                    svc = DetectionService.__new__(DetectionService)
                    svc.settings = settings
                    svc._record_alarm = Mock(return_value={"active": False})
                    det = svc._finish_detect(res, np.zeros((300, 290, 3), np.uint8), "loc.png", "test",
                                             time.perf_counter(), persist=False, with_heatmap=True)
                    # 渲染框不再覆盖引擎框；热图框单独以 heatmap_cc / visual_only 记录
                    self.assertEqual(det["defect_boxes"][0], [33, 33, 34, 34])
                    loc = det["n_tiles"]["localization"]
                    self.assertEqual(loc["regions"][0]["region_id"], "r0")
                    self.assertTrue(loc["heatmap_boxes"])
                    self.assertTrue(all(h["source"] == "heatmap_cc" and h["status"] == "visual_only"
                                        for h in loc["heatmap_boxes"]))
                    self.assertTrue(Path(det["overlay_path"]).exists())
                    did = DetectionService._persist(Mock(), det, None, "image")
                    with database.session_scope() as s:
                        d = to_dict(s.get(Detection, did))
                        self.assertEqual(d["localization"]["regions"][0]["source"], "engine_box")
                        self.assertIsNone(d["region_review"])

                    app = FastAPI()
                    app.add_middleware(ApiKeyMiddleware)
                    router = APIRouter()
                    register(router)
                    app.include_router(router, prefix="/api")
                    url = f"/api/detections/{did}/regions"
                    op_h, eng_h = {"X-API-Key": "t-op"}, {"X-API-Key": "t-eng"}
                    payload = {"op": "confirm", "region_id": "r0", "operator": "工程师甲",
                               "event_id": "ev1", "expected_revision": 0}
                    with TestClient(app) as c:
                        got = c.get(url, headers=op_h).json()
                        self.assertEqual((got["revision"], len(got["regions"]), got["coord_space"]), (0, 2, [290, 300]))
                        self.assertEqual(c.post(url, headers=op_h, json=payload).status_code, 403)
                        first = c.post(url, headers=eng_h, json=payload)
                        self.assertEqual(first.status_code, 200, first.text)
                        self.assertFalse(first.json()["idempotent"])
                        self.assertTrue(c.post(url, headers=eng_h, json=payload).json()["idempotent"])
                        self.assertEqual(c.post(url, headers=eng_h, json={**payload, "region_id": "r1"}).status_code, 409)
                        self.assertEqual(c.post(url, headers=eng_h, json={**payload, "event_id": "ev2"}).status_code, 409)
                        bad = {**payload, "op": "add", "region_id": None, "bbox": [0, 0, 999, 9],
                               "event_id": "ev3", "expected_revision": 1}
                        self.assertEqual(c.post(url, headers=eng_h, json=bad).status_code, 422)
                        miss = {**payload, "region_id": "zz", "event_id": "ev4", "expected_revision": 1}
                        self.assertEqual(c.post(url, headers=eng_h, json=miss).status_code, 404)
                        add = {**bad, "bbox": [1, 1, 20, 20]}
                        r = c.post(url, headers=eng_h, json=add)
                        self.assertEqual(r.status_code, 200, r.text)
                        self.assertEqual(r.json()["review"]["revision"], 2)
                        self.assertEqual(c.get(f"/api/detections/{did + 99}/regions", headers=op_h).status_code, 404)
                    with database.session_scope() as s:
                        nt = s.get(Detection, did).n_tiles
                        self.assertEqual(nt["region_review"]["revision"], 2)
                        self.assertEqual(len(L.trainable_region_boxes(nt["region_review"])), 2)
                        self.assertIn("localization", nt)  # 复核不破坏原定位记录
                        self.assertEqual(to_dict(s.get(Detection, did))["region_review"]["revision"], 2)
            finally:
                engine.dispose()


if __name__ == "__main__":
    unittest.main(verbosity=2)
