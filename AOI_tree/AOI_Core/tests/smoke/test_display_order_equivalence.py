"""Display-only channel permutation must preserve complete detection outputs."""
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.db import database
from backend.db.models import Base, Detection, StatsDaily
from backend.engine import DetectionResult
from backend.pipeline import service as module


def _legacy_display_bgr(arr_rgb, disp_wh):
    """旧顺序：先换通道再缩放。"""
    img_bgr = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)
    if (int(disp_wh[0]), int(disp_wh[1])) != (img_bgr.shape[1], img_bgr.shape[0]):
        img_bgr = cv2.resize(img_bgr, (int(disp_wh[0]), int(disp_wh[1])),
                             interpolation=cv2.INTER_AREA)
    return img_bgr


def legacy_finish():
    def run(self, *args, **kwargs):
        with patch.object(module, "_display_bgr", _legacy_display_bgr):
            return module.DetectionService._finish_detect(self, *args, **kwargs)
    return run


class DisplayOrderEquivalenceTest(unittest.TestCase):
    def test_channel_permutation_is_exact(self):
        rng = np.random.default_rng(42)
        for shape, size in [((2500, 2500, 3), (1280, 1280)),
                            ((71, 97, 3), (47, 34)), ((100, 150, 3), (50, 33))]:
            rgb = rng.integers(0, 256, shape, dtype=np.uint8)
            old = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), size,
                             interpolation=cv2.INTER_AREA)
            new = cv2.cvtColor(cv2.resize(rgb, size, interpolation=cv2.INTER_AREA),
                               cv2.COLOR_RGB2BGR)
            np.testing.assert_array_equal(old, new)

    def test_complete_output_and_database_contract(self):
        rng = np.random.default_rng(42)
        with tempfile.TemporaryDirectory(prefix="aoi_display_contract_") as tmp:
            root = Path(tmp)
            engine = create_engine("sqlite:///" + (root / "contract.db").as_posix())
            Base.metadata.create_all(engine)
            factory = sessionmaker(bind=engine, expire_on_commit=False)
            settings = SimpleNamespace(get=lambda *args: 128,
                section=lambda key: {"enabled": False},
                storage=lambda key: root)
            service = module.DetectionService.__new__(module.DetectionService)
            service.settings = settings
            result = DetectionResult(score=0.83, decision="anomaly", is_anomaly=True,
                threshold=0.7, model_version=1, slot_scores={"sem": 0.83},
                types=[{"type": "scratch", "confidence": 0.9}],
                align_offset=[1.25, -2.0], align_warn=True,
                mask=rng.random((32, 32)).astype(np.float32))
            old_finish = legacy_finish()
            try:
                with patch.object(database, "_engine", engine), \
                     patch.object(database, "_SessionLocal", factory), \
                     patch.object(service, "_record_alarm", return_value={"active": True, "window_hits": 5}), \
                     patch.object(module, "notify_event"), patch.object(module.time, "time", return_value=1):
                    for shape in ((250, 250, 3), (47, 73, 3), (201, 313, 3)):
                        rgb = rng.integers(0, 256, shape, dtype=np.uint8)
                        old = old_finish(service, result, rgb, "input.jpg", "test", time.perf_counter())
                        old_heat = Path(old["heatmap_path"]).read_bytes()
                        old_overlay = Path(old["overlay_path"]).read_bytes()
                        new = service._finish_detect(result, rgb, "input.jpg", "test", time.perf_counter())
                        self.assertEqual(old_heat, Path(new["heatmap_path"]).read_bytes())
                        self.assertEqual(old_overlay, Path(new["overlay_path"]).read_bytes())
                        for key in old:
                            if key not in ("total_latency_ms", "latency_e2e_ms", "detection_id"):
                                self.assertEqual(old[key], new[key], key)
                        with database.session_scope() as session:
                            a = session.get(Detection, old["detection_id"])
                            b = session.get(Detection, new["detection_id"])
                            self.assertEqual(a.final_score, b.final_score)
                            self.assertEqual(a.is_anomaly, b.is_anomaly)
                            self.assertEqual({k: v for k, v in a.n_tiles.items() if k != "total_latency_ms"},
                                             {k: v for k, v in b.n_tiles.items() if k != "total_latency_ms"})
                            self.assertEqual(a.heatmap_path, b.heatmap_path)
                            self.assertEqual(a.overlay_path, b.overlay_path)
                    with database.session_scope() as session:
                        self.assertEqual(session.query(Detection).count(), 6)
                        daily = session.query(StatsDaily).first()
                        self.assertEqual((daily.n_inspected, daily.n_anomaly), (6, 6))
            finally:
                engine.dispose()


if __name__ == "__main__":
    unittest.main()
