"""独立配对热图 API：mock 路径不加载 DINO、不写 DB。"""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

from backend.api.app import create_app


class PairHeatmapApiTest(unittest.TestCase):
    def _request(self, directory, image_shape=(8, 8, 3)):
        p = Path(directory) / "x.png"
        cv2.imwrite(str(p), np.zeros(image_shape, np.uint8))
        return TestClient(create_app()), p

    def test_success_response_fields_are_isolated(self):
        with TemporaryDirectory() as d:
            client, p = self._request(d)
            raw = np.ones((2, 2), np.float32)
            backbone = type("Backbone", (), {"extract_tiles": lambda self, images: object()})()
            slot = type("Slot", (), {"score_image": lambda self, *a, **k: (0.25, [], [raw])})()
            with patch("backend.api.routes_pair_heatmap._pair_engine", return_value=(backbone, slot)), \
                 patch("backend.api.routes_pair_heatmap.get_settings") as settings:
                settings.return_value.storage.return_value = Path(d)
                with patch("backend.api.routes_pair_heatmap.render_heatmap_overlay",
                           return_value=(np.zeros((8, 8, 3), np.uint8),
                                         np.zeros((8, 8, 3), np.uint8), [])):
                    r = client.post("/api/detect/pair-heatmap", json={
                        "path": str(p), "pair_template": str(p), "same_view_confirmed": True})
            self.assertEqual(r.status_code, 200)
            body = r.json()
            self.assertEqual(body["status"], "ok")
            self.assertEqual(body["same_view_confirmed"], True)
            self.assertEqual(body["aux_pair"]["visual_only"], True)
            self.assertNotIn("boxes", body["aux_pair"])
            self.assertTrue(Path(body["raw_path"]).is_file())

    def test_confirmation_and_validation_before_model(self):
        with TemporaryDirectory() as d:
            p = Path(d) / "x.png"
            cv2.imwrite(str(p), np.zeros((8, 8, 3), np.uint8))
            client = TestClient(create_app())
            with patch("backend.api.routes_pair_heatmap._pair_engine",
                       side_effect=AssertionError("must not load model")):
                r = client.post("/api/detect/pair-heatmap", json={
                    "path": str(p), "pair_template": str(p),
                    "same_view_confirmed": False})
            self.assertEqual(r.status_code, 400)
            self.assertIn("same_view_confirmed", r.json()["detail"])

    def test_invalid_file_is_400_without_model(self):
        client = TestClient(create_app())
        with patch("backend.api.routes_pair_heatmap._pair_engine",
                   side_effect=AssertionError("must not load model")):
            r = client.post("/api/detect/pair-heatmap", json={
                "path": "missing.png", "pair_template": "missing-template.png",
                "same_view_confirmed": True})
        self.assertEqual(r.status_code, 400)

    def test_size_mismatch_is_400_without_model(self):
        with TemporaryDirectory() as d:
            client, p = self._request(d, (8, 8, 3))
            q = Path(d) / "template.png"
            cv2.imwrite(str(q), np.zeros((7, 8, 3), np.uint8))
            with patch("backend.api.routes_pair_heatmap._pair_engine",
                       side_effect=AssertionError("must not load model")):
                r = client.post("/api/detect/pair-heatmap", json={
                    "path": str(p), "pair_template": str(q), "same_view_confirmed": True})
        self.assertEqual(r.status_code, 400)
        self.assertIn("尺寸不符", r.json()["detail"])

    def test_raw_write_failure_is_500(self):
        with TemporaryDirectory() as d:
            client, p = self._request(d)
            backbone = type("Backbone", (), {"extract_tiles": lambda self, images: object()})()
            slot = type("Slot", (), {"score_image": lambda self, *a, **k: (0.25, [], [np.ones((2, 2), np.float32)])})()
            with patch("backend.api.routes_pair_heatmap._pair_engine", return_value=(backbone, slot)), \
                 patch("backend.api.routes_pair_heatmap.get_settings") as settings, \
                 patch("backend.api.routes_pair_heatmap._save_npy", return_value=False):
                settings.return_value.storage.return_value = Path(d)
                r = client.post("/api/detect/pair-heatmap", json={
                    "path": str(p), "pair_template": str(p), "same_view_confirmed": True})
        self.assertEqual(r.status_code, 500)
        self.assertFalse(any(Path(d).glob("*_raw.npy")))

    def test_cuda_unavailable_is_503_without_db(self):
        with TemporaryDirectory() as d:
            p = Path(d) / "x.png"
            cv2.imwrite(str(p), np.zeros((8, 8, 3), np.uint8))
            client = TestClient(create_app())
            with patch("backend.api.routes_pair_heatmap.torch.cuda.is_available",
                       return_value=False):
                r = client.post("/api/detect/pair-heatmap", json={
                    "path": str(p), "pair_template": str(p),
                    "same_view_confirmed": True})
            self.assertEqual(r.status_code, 503)
            self.assertIn("CUDA", r.json()["detail"])


if __name__ == "__main__":
    unittest.main()
