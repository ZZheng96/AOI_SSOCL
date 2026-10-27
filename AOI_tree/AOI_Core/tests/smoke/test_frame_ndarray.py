"""§77 流/视频帧内存直检：detect_ndarray 契约，以及 RTSP/视频抽帧不再走临时 JPEG 往返。

不加载模型：引擎用假对象记录送检数组，DB 用临时 sqlite。
"""
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.db import database
from backend.db.models import Base, Detection, Image as ImageRow
from backend.engine import DetectionResult
from backend.pipeline import service as module
from backend.pipeline.pipeline_service import PipelineService

VOLATILE = {"latency_e2e_ms", "total_latency_ms"}


class _FakeEngine:
    def __init__(self, decision="normal"):
        self.calls = []
        self.decision = decision

    def predict_ndarray(self, category, rgb, path_hint=None):
        self.calls.append((rgb.copy(), path_hint))
        score = float(rgb.mean()) / 255.0
        return DetectionResult(score=score, decision=self.decision,
                               is_anomaly=self.decision == "anomaly",
                               threshold=0.7, model_version=1,
                               slot_scores={"sem": score})


def _frames(n=3, shape=(48, 64, 3)):
    rng = np.random.default_rng(77)
    return [rng.integers(0, 256, shape, dtype=np.uint8) for _ in range(n)]


def _write_video(path, frames):
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()


class _Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="aoi_frame_nd_")
        self.root = Path(self._tmp.name)
        self.db = create_engine("sqlite:///" + (self.root / "t.db").as_posix())
        Base.metadata.create_all(self.db)
        factory = sessionmaker(bind=self.db, expire_on_commit=False)
        self.fake = _FakeEngine()
        self.svc = module.DetectionService.__new__(module.DetectionService)
        self.archive_cfg = {"enabled": False}
        self.svc.settings = SimpleNamespace(
            get=lambda sec, key, default=None: {"video_frame_interval": 1}.get(key, default),
            section=lambda key: self.archive_cfg,
            storage=lambda key: self.root / key)
        for key in ("uploads", "archive", "heatmaps"):
            (self.root / key).mkdir()
        self.patches = [patch.object(database, "_engine", self.db),
                        patch.object(database, "_SessionLocal", factory),
                        patch.object(module, "get_engine", return_value=self.fake),
                        patch.object(module, "notify_event"),
                        patch.object(self.svc, "_record_alarm",
                                     return_value={"active": False, "window_hits": 0})]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.db.dispose()
        self._tmp.cleanup()


class DetectNdarrayTest(_Base):
    def test_equivalent_to_detect_frame(self):
        bgr = _frames(1)[0]
        ok, png = cv2.imencode(".png", bgr)          # 无损：两路输入像素一致
        a = self.svc.detect_frame(png.tobytes(), "test", persist=False, with_heatmap=False)
        b = self.svc.detect_ndarray(bgr, "test", target_type="upload",
                                    persist=False, with_heatmap=False)
        self.assertEqual({k: v for k, v in a.items() if k not in VOLATILE},
                         {k: v for k, v in b.items() if k not in VOLATILE})
        self.assertTrue(a["zero_copy"] and b["zero_copy"])
        self.assertEqual(a["image_path"], "<memory_frame>")
        (ra, ha), (rb, hb) = self.fake.calls
        np.testing.assert_array_equal(ra, rb)
        np.testing.assert_array_equal(ra, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        self.assertIsNone(ha)
        self.assertIsNone(hb)

    def test_persist_without_file_uses_memory_path_and_archives_frame(self):
        self.fake.decision = "anomaly"
        self.archive_cfg = {"enabled": True}
        bgr = _frames(1)[0]
        with patch("backend.core.datasets.get_or_create_dataset", return_value=999):
            out = self.svc.detect_ndarray(bgr, "test", target_type="stream",
                                          persist=True, with_heatmap=False,
                                          workorder_id=None)
        self.assertTrue(out["persisted"])
        with database.session_scope() as s:
            row = s.get(Detection, out["detection_id"])
            self.assertEqual((row.image_path, row.target_type), ("<memory_frame>", "stream"))
            img = s.query(ImageRow).filter(ImageRow.source == "auto_archive").one()
            arch = Path(img.path)
        self.assertEqual(Path(out["archive_path"]), arch)
        self.assertTrue(arch.is_file())
        back = cv2.imread(str(arch))
        self.assertEqual(back.shape, bgr.shape)

    def test_memory_frame_heatmap_filename_is_valid(self):
        orig = self.fake.predict_ndarray

        def with_hm(category, rgb, path_hint=None):
            r = orig(category, rgb, path_hint)
            r.fused_heatmap = np.random.default_rng(1).random(rgb.shape[:2]).astype(np.float32)
            return r

        self.fake.predict_ndarray = with_hm
        ok, png = cv2.imencode(".png", _frames(1)[0])
        out = self.svc.detect_frame(png.tobytes(), "test", persist=False, with_heatmap=True)
        for key in ("heatmap_path", "overlay_path"):
            p = Path(out[key])
            self.assertNotRegex(p.name, r'[<>:"|?*]')
            module.wait_artifact(str(p), timeout=10)
            self.assertTrue(p.is_file(), key)

    def test_detect_frame_does_not_archive_memory(self):
        self.fake.decision = "anomaly"
        self.archive_cfg = {"enabled": True}
        ok, png = cv2.imencode(".png", _frames(1)[0])
        out = self.svc.detect_frame(png.tobytes(), "test", persist=True, with_heatmap=False)
        self.assertTrue(out["persisted"])
        self.assertNotIn("archive_path", out)       # 行为同改造前：无源文件不归档


class VideoFramePathTest(_Base):
    def test_video_uses_raw_frame_and_keeps_frame_file(self):
        frames = _frames(3)
        vid = self.root / "v.avi"
        _write_video(vid, frames)
        cap = cv2.VideoCapture(str(vid))
        decoded = [cap.read()[1] for _ in range(3)]
        cap.release()
        seen = []

        def register(video_id, idx, path, category, ts):
            seen.append(Path(path).is_file())        # 登记时帧图必须已落盘
            return None

        with patch.object(self.svc, "detect_image",
                          side_effect=AssertionError("不应再走 detect_image")), \
                patch.object(self.svc, "_register_frame", side_effect=register):
            dets = list(self.svc.iter_video_detections(vid, "test", interval=1,
                                                       video_id=5, persist=True,
                                                       with_heatmap=False))
        self.assertEqual(len(dets), 3)
        self.assertEqual(seen, [True, True, True])
        for i, (det, (rgb, hint)) in enumerate(zip(dets, self.fake.calls)):
            tmp = self.root / "uploads" / f"frame_5_{i}.jpg"
            self.assertEqual(det["image_path"], str(tmp))
            self.assertEqual(hint, str(tmp))
            self.assertEqual(det["frame_idx"], i)
            # 送检的是原始解码帧，而非 JPEG 重压缩后读回的像素
            np.testing.assert_array_equal(rgb, cv2.cvtColor(decoded[i], cv2.COLOR_BGR2RGB))
            with database.session_scope() as s:
                self.assertEqual(s.get(Detection, det["detection_id"]).image_path, str(tmp))


class RtspLoopTest(unittest.TestCase):
    def test_rtsp_loop_calls_detect_ndarray_without_tempfile(self):
        frames = _frames(2)
        with tempfile.TemporaryDirectory(prefix="aoi_rtsp_") as d:
            vid = Path(d) / "s.avi"
            _write_video(vid, frames)
            ps = PipelineService(interval=0)
            stop = threading.Event()
            calls = []

            def detect_ndarray(frame, category, **kw):
                calls.append((frame.shape, category, kw))
                if len(calls) >= 3:
                    stop.set()
                return {}

            svc = Mock()
            svc.detect_ndarray.side_effect = detect_ndarray
            svc.detect_image.side_effect = AssertionError("不应再走 detect_image")
            src = SimpleNamespace(id=1, stream_config={"path": str(vid), "category": "pcb"})
            with patch("backend.pipeline.service.get_detection_service", return_value=svc), \
                    patch("tempfile.mkstemp", side_effect=AssertionError("不应建临时文件")), \
                    patch("cv2.imwrite", side_effect=AssertionError("不应写临时 JPEG")):
                guard = threading.Timer(5, stop.set)   # 兜底防死循环
                guard.start()
                t0 = time.time()
                ps._rtsp_loop(src, workorder_id=9, stop_event=stop)
                guard.cancel()
                self.assertLess(time.time() - t0, 10)
        self.assertEqual(len(calls), 3)               # 本地视频循环模拟流（2 帧后回绕）
        svc.detect_image.assert_not_called()
        for shape, cat, kw in calls:
            self.assertEqual(shape, frames[0].shape)
            self.assertEqual(cat, "pcb")
            self.assertEqual(kw, {"persist": True, "with_heatmap": True,
                                  "target_type": "stream", "workorder_id": 9})


if __name__ == "__main__":
    unittest.main()
