"""AC-063：原始浮点热图留存 + 离线按阈值复算 mask/框 + 孤儿/白名单/穿越。"""
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from backend.core.imaging import imread
from backend.pipeline import service
from backend.pipeline.service import (DetectionService, load_raw_heatmap,
                                      quantize_raw_heatmap, recompute_from_postproc,
                                      recompute_heatmap_mask)


def _fake_result(hm):
    return SimpleNamespace(
        slot_scores={"sem": 0.9}, raw_scores={}, weights={}, router_w={}, extra={},
        triggered_slot="sem", decision="anomaly", score=0.9, is_anomaly=True,
        threshold=0.5, gray_threshold=0.4, open_alert=False, types=[],
        align_offset=None, align_warn=False, model_version="t", latency_ms=1.0,
        defect_boxes=[{"bbox": [100, 100, 300, 300]}], fused_heatmap=hm, mask=None)


def _heatmap(h=384, w=512, seed=0):
    rng = np.random.default_rng(seed)
    hm = rng.random((h, w)).astype(np.float32) * 0.3
    hm[h * 60 // 384:h * 140 // 384, w * 80 // 512:w * 200 // 512] += 1.7   # 主异常块
    hm[h * 250 // 384:h * 270 // 384, w * 400 // 512:w * 430 // 512] += 1.1  # 次块（阈值敏感）
    hm[h * 300 // 384, w * 50 // 512] += 2.0                                 # 单像素峰
    return hm


class RawHeatmapRetentionTest(unittest.TestCase):
    def _run(self, tmp, overrides):
        settings = Mock()
        settings.get.side_effect = lambda sec, key, default=None: overrides.get(key, default)
        settings.storage.return_value = Path(tmp)
        svc = DetectionService.__new__(DetectionService)
        svc.settings = settings
        svc._record_alarm = Mock(return_value={"active": False})
        rgb = np.zeros((1500, 2000, 3), np.uint8)
        with patch("backend.pipeline.service.notify_event"):
            return svc._finish_detect(_fake_result(_heatmap()), rgb, "raw.png", "test",
                                      time.perf_counter(), persist=False, with_heatmap=True)

    def test_saved_raw_and_default_recompute_matches_online(self):
        with tempfile.TemporaryDirectory() as tmp:
            det = self._run(tmp, {"overlay_max_dim": 1280})
            pp = det["n_tiles"]["localization"]["heatmap_postproc"]
            json.dumps(det, allow_nan=False)
            raw_path = Path(det["raw_heatmap_path"])
            self.assertTrue(raw_path.is_file())
            self.assertEqual(pp["raw_heatmap_path"], str(raw_path))
            self.assertEqual(pp["raw_shape"], [384, 512])
            self.assertEqual(pp["raw_dtype"], "float16")
            self.assertEqual(pp["display_size"], [1280, 960])
            self.assertEqual(pp["image_size"], [2000, 1500])
            self.assertAlmostEqual(pp["display_scale"], 0.64)
            self.assertEqual(pp["scale_raw_to_display"], [2.5, 2.5])
            self.assertAlmostEqual(pp["scale_raw_to_image"][0], 2000 / 512)
            raw = np.load(raw_path, allow_pickle=False)
            self.assertEqual((raw.dtype, raw.shape), (np.float16, (384, 512)))
            # 默认参数复算：mask 与在线留存 PNG 逐像素一致，框与在线 heatmap_boxes 一致
            rec = recompute_from_postproc(pp, tmp)
            online_mask = imread(det["mask_path"], 0)
            self.assertTrue(np.array_equal(rec["mask"], online_mask))
            online_boxes = [h["bbox"] for h in det["n_tiles"]["localization"]["heatmap_boxes"]]
            self.assertTrue(online_boxes)
            self.assertEqual(rec["boxes_image"], online_boxes)
            # 改阈值：更低阈值框面积不减，高阈值丢掉次块
            low = recompute_from_postproc(pp, tmp, rel_thresh=0.3)
            high = recompute_from_postproc(pp, tmp, rel_thresh=0.8)
            self.assertGreaterEqual(int((low["mask"] > 0).sum()), int((rec["mask"] > 0).sum()))
            self.assertLess(len(high["boxes_image"]), len(rec["boxes_image"]))

    def test_switch_off_writes_no_raw(self):
        with tempfile.TemporaryDirectory() as tmp:
            det = self._run(tmp, {"overlay_max_dim": 1280, "overlay_save_raw_heatmap": False})
            pp = det["n_tiles"]["localization"]["heatmap_postproc"]
            self.assertNotIn("raw_heatmap_path", det)
            self.assertNotIn("raw_heatmap_path", pp)
            self.assertEqual(list(Path(tmp).glob("*_raw.npy")), [])
            self.assertTrue(Path(det["overlay_path"]).is_file())

    def test_pure_recompute_equals_render(self):
        img = np.zeros((96, 128, 3), np.uint8)
        hm = quantize_raw_heatmap(_heatmap(48, 64, seed=3))
        _, _, boxes, mask = service.render_heatmap_overlay(img, hm, return_mask=True)
        rec = recompute_heatmap_mask(hm, (128, 96))
        self.assertTrue(np.array_equal(rec["mask"], mask))
        self.assertEqual(rec["boxes_display"], boxes)
        # 非有限值不量化为 float16（保留 float32）
        big = np.array([[1e6, 0.0]], np.float32)
        self.assertEqual(quantize_raw_heatmap(big).dtype, np.float32)

    def test_load_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "heatmaps"
            base.mkdir()
            np.save(str(Path(tmp) / "evil.npy"), np.zeros((2, 2), np.float16))
            np.save(str(base / "ok_raw.npy"), np.zeros((2, 2), np.float16))
            self.assertEqual(load_raw_heatmap(base / "ok_raw.npy", base).shape, (2, 2))
            self.assertEqual(load_raw_heatmap("ok_raw.npy", base).shape, (2, 2))
            for bad in ("../evil.npy", str(Path(tmp) / "evil.npy"), "ok_raw.txt"):
                with self.assertRaises(PermissionError):
                    load_raw_heatmap(bad, base)

    def test_orphan_refs_and_file_whitelist(self):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from backend.api import routes_data
        from backend.core import security
        from backend.db import database
        from backend.db.models import Base, Detection
        with tempfile.TemporaryDirectory() as td:
            eng = create_engine(f"sqlite:///{Path(td) / 't.db'}")
            try:
                Base.metadata.create_all(eng)
                SL = sessionmaker(bind=eng)
                hm_dir = Path(td).resolve() / "moved" / "heatmaps"
                hm_dir.mkdir(parents=True)
                ov = str(hm_dir / "a_1_overlay.jpg")
                fake = SimpleNamespace(storage_dir=Path(td) / "storage",
                                       engine_storage=Path(td) / "engine")
                with patch.object(database, "_engine", eng), \
                        patch.object(database, "_SessionLocal", SL), \
                        patch.object(security, "get_settings", return_value=fake):
                    s = SL()
                    s.add(Detection(image_path="x.jpg", final_score=0.1, is_anomaly=False,
                                    latency_ms=1.0, overlay_path=ov))
                    s.commit()
                    s.close()
                    refs = routes_data._referenced_paths()
                    # 存储迁移后（不在 storage_dir 内）mask/raw 仍凭 overlay 登记放行
                    self.assertTrue(security.is_path_allowed(str(hm_dir / "a_1_raw.npy")))
                    self.assertTrue(security.is_path_allowed(str(hm_dir / "a_1_mask.png")))
                    self.assertFalse(security.is_path_allowed(str(hm_dir / "b_2_raw.npy")))
                    self.assertFalse(security.is_path_allowed(
                        str(hm_dir / ".." / "a_1_raw.npy")))
                    self.assertFalse(security.is_path_allowed(str(hm_dir / "a_1_heat.npy")))
            finally:
                eng.dispose()
            self.assertIn(str(hm_dir / "a_1_raw.npy"), refs)
            self.assertIn(str(hm_dir / "a_1_mask.png"), refs)


if __name__ == "__main__":
    unittest.main()
