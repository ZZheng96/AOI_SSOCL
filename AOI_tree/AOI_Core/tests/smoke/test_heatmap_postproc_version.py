"""AC-065：热图后处理参数可配置 + 确定性 version_id。"""
import unittest

import numpy as np

from backend.pipeline.service import heatmap_postproc_meta, render_heatmap_overlay


class HeatmapPostprocVersionTest(unittest.TestCase):
    def test_version_deterministic_and_param_sensitive(self):
        a = heatmap_postproc_meta(0.6, 16, 3, 1.0)
        b = heatmap_postproc_meta(0.6, 16, 3, 0.5)
        c = heatmap_postproc_meta(0.7, 16, 3, 1.0)
        d = heatmap_postproc_meta(0.6, 16, 1, 1.0)
        self.assertEqual(a["version_id"], b["version_id"])  # 显示缩放不影响版本
        self.assertNotEqual(a["version_id"], c["version_id"])
        self.assertNotEqual(a["version_id"], d["version_id"])
        for k in ("algo", "threshold_basis", "min_area_unit", "morph_kernel"):
            self.assertIn(k, a)

    def test_morph_kernel_and_min_area_effect(self):
        img = np.zeros((64, 64, 3), np.uint8)
        hm = np.zeros((64, 64), np.float32)
        hm[10:30, 10:30] = 1.0     # 大块 400px
        hm[50, 50] = 1.0           # 单像素噪点
        _, _, boxes_open = render_heatmap_overlay(img, hm, min_area=1, morph_kernel=3)
        _, _, boxes_raw = render_heatmap_overlay(img, hm, min_area=1, morph_kernel=1)
        self.assertEqual(len(boxes_open), 1)
        self.assertEqual(len(boxes_raw), 2)
        _, _, boxes_big = render_heatmap_overlay(img, hm, min_area=500, morph_kernel=3)
        self.assertEqual(boxes_big, [])

    def test_return_mask_binary_matches_boxes(self):
        img = np.zeros((64, 64, 3), np.uint8)
        hm = np.zeros((64, 64), np.float32)
        hm[10:30, 10:30] = 1.0
        hm[50, 50] = 1.0
        _, _, boxes, mask = render_heatmap_overlay(img, hm, min_area=1,
                                                   morph_kernel=3, return_mask=True)
        self.assertEqual(mask.shape, (64, 64))
        self.assertEqual(set(np.unique(mask).tolist()), {0, 255})
        self.assertEqual(mask[50, 50], 0)        # 开运算去噪后的 mask
        self.assertEqual(int((mask > 0).sum()), 400)
        self.assertEqual(len(boxes), 1)

    def test_orphan_scan_keeps_mask_sibling(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from backend.db import database
        from backend.db.models import Base, Detection
        from backend.api import routes_data
        with tempfile.TemporaryDirectory() as td:
            eng = create_engine(f"sqlite:///{Path(td) / 't.db'}")
            Base.metadata.create_all(eng)
            SL = sessionmaker(bind=eng)
            with patch.object(database, "_engine", eng), \
                    patch.object(database, "_SessionLocal", SL):
                s = SL()
                ov = str(Path(td) / "a_1_overlay.jpg")
                s.add(Detection(image_path="x.jpg", final_score=0.1, is_anomaly=False,
                                latency_ms=1.0, overlay_path=ov))
                s.commit(); s.close()
                refs = routes_data._referenced_paths()
            eng.dispose()
            self.assertIn(str(Path(td) / "a_1_mask.png"), refs)


if __name__ == "__main__":
    unittest.main()
