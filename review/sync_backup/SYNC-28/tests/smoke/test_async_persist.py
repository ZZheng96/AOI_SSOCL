"""TEST-01 dev_async：热图着色/叠加/编码移入线程池后的契约。

1) 文件最终写出，且与同步 render_heatmap_overlay 逐字节一致；
2) 写盘失败被记录（日志 + Detection.n_tiles["artifact_error"]），不抛异常；
3) 响应字段（键集与可视化相关值）不变；
4) /api/file 在产物写盘在途时等待，超时 503，失败 500。
"""
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.api import routes_system
from backend.db import database
from backend.db.models import Base, Detection
from backend.engine import DetectionResult
from backend.pipeline import service as module

EXPECTED_KEYS = {
    "image_path", "category", "final_score", "is_anomaly", "decision", "threshold",
    "gray_threshold", "level1_score", "level2_score", "level3_score", "triggered_level",
    "slots", "weights", "raw_scores", "cal_scores", "router_w", "actual_router",
    "candidate_router", "feature_metadata", "triggered_slot", "box_evidence",
    "localization", "open_alert", "e_open_score", "e_open_threshold", "defect_types",
    "align_offset", "align_warn", "model_version", "latency_ms", "total_latency_ms",
    "lock_wait_ms", "decision_trace", "n_tiles", "defect_boxes", "heatmap_path", "overlay_path",
    "mask_path", "raw_heatmap_path", "latency_e2e_ms", "persisted",
    "feedback_supported", "detection_id", "alarm",
    "final_status", "is_defect", "requires_review", "is_system_error"}


class AsyncPersistTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="aoi_async_persist_")
        self.root = Path(self._tmp.name)
        self.engine = create_engine("sqlite:///" + (self.root / "t.db").as_posix())
        Base.metadata.create_all(self.engine)
        factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        overrides = {"overlay_max_dim": 1280}
        self.svc = module.DetectionService.__new__(module.DetectionService)
        self.svc.settings = SimpleNamespace(
            get=lambda sec, key, default=None: overrides.get(key, default),
            section=lambda key: {"enabled": False},
            storage=lambda key: self.root)
        self.patches = [patch.object(database, "_engine", self.engine),
                        patch.object(database, "_SessionLocal", factory),
                        patch.object(self.svc, "_record_alarm",
                                     return_value={"active": False, "window_hits": 0}),
                        patch.object(module, "notify_event")]
        for p in self.patches:
            p.start()
        self.rng = np.random.default_rng(7)

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.engine.dispose()
        self._tmp.cleanup()

    def _result(self):
        return DetectionResult(
            score=0.9, decision="anomaly", is_anomaly=True, threshold=0.7,
            model_version=1, slot_scores={"sem": 0.9},
            defect_boxes=[{"bbox": [40, 50, 400, 600], "dominant_slot": "sem"}],
            fused_heatmap=self.rng.random((48, 48)).astype(np.float32))

    def _run(self, shape, name="img.jpg"):
        rgb = self.rng.integers(0, 256, shape, dtype=np.uint8)
        res = self._result()
        out = self.svc._finish_detect(res, rgb, name, "test", time.perf_counter())
        return rgb, res, out

    def test_files_written_and_identical_to_sync_render(self):
        for shape in ((2500, 2500, 3), (300, 290, 3)):
            rgb, res, out = self._run(shape)
            for key in ("heatmap_path", "overlay_path", "mask_path", "raw_heatmap_path"):
                self.assertTrue(Path(out[key]).is_file(), key)
                self.assertIsNone(module.wait_artifact(out[key]), key)  # 已出在途表
            # 同步参考：旧路径（先换通道再缩放 + render_heatmap_overlay）
            h0, w0 = shape[:2]
            scale = min(1.0, 1280 / max(h0, w0))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if scale < 1.0:
                bgr = cv2.resize(bgr, (int(w0 * scale), int(h0 * scale)),
                                 interpolation=cv2.INTER_AREA)
            hm = module.quantize_raw_heatmap(res.fused_heatmap)
            eng = [[v * scale for v in b["bbox"]] for b in res.defect_boxes]
            heat, ov, boxes, mask = module.render_heatmap_overlay(
                bgr, hm, engine_boxes=eng, return_mask=True)
            for arr, key in ((heat, "heatmap_path"), (ov, "overlay_path"),
                             (mask, "mask_path")):
                ext = Path(out[key]).suffix
                self.assertEqual(cv2.imencode(ext, arr)[1].tobytes(),
                                 Path(out[key]).read_bytes(), key)
            if scale < 1.0:
                boxes = [[int(x / scale), int(y / scale), int(w / scale), int(h / scale)]
                         for x, y, w, h in boxes]
            self.assertEqual([b["bbox"] for b in out["localization"]["heatmap_boxes"]],
                             [[x, y, x + w, y + h] for x, y, w, h in boxes])
            self.assertEqual(out["localization"]["heatmap_postproc"]["display_size"],
                             [bgr.shape[1], bgr.shape[0]])

    def test_response_fields_unchanged(self):
        _, _, out = self._run((300, 290, 3))
        self.assertEqual(set(out), EXPECTED_KEYS)
        self.assertTrue(out["persisted"])
        with database.session_scope() as s:
            row = s.get(Detection, out["detection_id"])
            self.assertNotIn("artifact_error", row.n_tiles)
            self.assertEqual(row.overlay_path, out["overlay_path"])

    def test_write_failure_logged_and_marked(self):
        real = module.imwrite

        def flaky(path, img):
            return False if str(path).endswith("_overlay.jpg") else real(path, img)

        with patch.object(module, "imwrite", flaky), \
                self.assertLogs(module.logger, level="WARNING") as logs:
            _, _, out = self._run((300, 290, 3))
        self.assertTrue(any("热力图写盘失败" in m and "overlay" in m for m in logs.output))
        self.assertTrue(out["persisted"])
        self.assertEqual(set(out), EXPECTED_KEYS)  # 响应字段不因失败而变
        self.assertFalse(Path(out["overlay_path"]).exists())
        self.assertTrue(Path(out["heatmap_path"]).is_file())
        with database.session_scope() as s:
            err = s.get(Detection, out["detection_id"]).n_tiles["artifact_error"]
        self.assertEqual(err["failed"], ["overlay"])

    def test_render_exception_is_contained(self):
        with patch.object(module, "_compose_overlay", side_effect=RuntimeError("boom")), \
                self.assertLogs(module.logger, level="WARNING"):
            _, _, out = self._run((300, 290, 3))
        with database.session_scope() as s:
            err = s.get(Detection, out["detection_id"]).n_tiles["artifact_error"]
        self.assertEqual(err["failed"], ["overlay"])

    def test_download_waits_for_pending_artifact(self):
        target = self.root / "pending_overlay.jpg"
        gate = threading.Event()

        def slow(path, img):
            gate.wait(5)
            return module.imwrite(path, img)

        ex = ThreadPoolExecutor(max_workers=1)
        try:
            with patch.object(module, "_io_exec", ex), \
                    patch("backend.core.security.is_path_allowed", return_value=True):
                fut = module._submit(slow, target, np.zeros((8, 8, 3), np.uint8))
                self.assertFalse(target.exists())
                threading.Timer(0.2, gate.set).start()
                resp = routes_system.api_file(str(target))  # 等待写完而非 404
                self.assertTrue(fut.done() and fut.result())
                self.assertEqual(Path(resp.path), target)
                # 超时 → 503；写盘失败 → 500
                with patch.object(module, "wait_artifact", side_effect=FutTimeout()):
                    with self.assertRaises(HTTPException) as cm:
                        routes_system.api_file(str(target))
                    self.assertEqual(cm.exception.status_code, 503)
                with patch.object(module, "wait_artifact", return_value=False):
                    with self.assertRaises(HTTPException) as cm:
                        routes_system.api_file(str(target))
                    self.assertEqual(cm.exception.status_code, 500)
        finally:
            gate.set()
            ex.shutdown(wait=True)


if __name__ == "__main__":
    unittest.main()
