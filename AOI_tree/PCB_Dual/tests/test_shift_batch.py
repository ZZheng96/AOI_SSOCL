from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from app.detect.adapters.shift_adapter import _load_shift


class ShiftBatchTests(unittest.TestCase):
    def setUp(self):
        self.alg = _load_shift()["instance"]
        self.image = np.full((48, 64, 3), 240, dtype=np.uint8)
        self.image[12:28, 18:38] = 10
        self.boxes = [(18, 12, 20, 16), (4, 7, 20, 16), (0, 0, 12, 10)]

    def assert_equal(self, a, b):
        self.assertEqual(a.code, b.code)
        self.assertEqual(a.message, b.message)
        self.assertEqual(len(a.parts), len(b.parts))
        for left, right in zip(a.parts, b.parts):
            for key in ("x", "y", "width", "height", "label", "angle", "confidence"):
                self.assertEqual(getattr(left, key, None), getattr(right, key, None))
        self.assertEqual(a.metadata.keys(), b.metadata.keys())
        for key in a.metadata:
            left, right = a.metadata[key], b.metadata[key]
            if isinstance(left, np.ndarray):
                np.testing.assert_array_equal(left, right, err_msg=key)
            else:
                self.assertEqual(left, right, key)

    def test_exact_output_and_diagnostics(self):
        for oob in (False, True):
            for diagnostics in (False, True):
                for rotation in (False, True):
                    with self.subTest(oob=oob, diagnostics=diagnostics, rotation=rotation):
                        cfg = dict(self.alg.default_config(), allow_out_of_bounds=oob,
                                   diagnostics=diagnostics, use_rotation=rotation)
                        expected = [self.alg.run(self.image, cfg, roi_bbox=b) for b in self.boxes]
                        actual = self.alg.run_many(self.image, cfg, self.boxes)
                        for a, b in zip(expected, actual):
                            self.assert_equal(a, b)

    def test_extract_and_integral_reuse(self):
        globals_ = self.alg.run.__func__.__globals__
        original = globals_["extract_base_mask"]
        for oob, count in ((False, 1), (True, 2)):
            cfg = dict(self.alg.default_config(), use_rotation=False, allow_out_of_bounds=oob)
            extract = Mock(wraps=original)
            with patch.dict(globals_, extract_base_mask=extract):
                with patch.object(cv2, "integral", wraps=cv2.integral) as integral:
                    self.alg.run_many(self.image, cfg, self.boxes)
                    self.assertEqual(extract.call_count, 1)
                    self.assertEqual(integral.call_count, count)

    def test_no_cross_request_or_config_reuse(self):
        cfg = dict(self.alg.default_config(), use_rotation=False)
        first = self.alg.run_many(self.image, cfg, self.boxes)
        bright = np.full_like(self.image, 240)
        second = self.alg.run_many(bright, cfg, self.boxes)
        self.assertFalse(np.array_equal(first[0].metadata["base_mask"], second[0].metadata["base_mask"]))
        cfg["feature_v_high"] = 255
        changed = self.alg.run_many(bright, cfg, self.boxes)
        self.assertFalse(np.array_equal(second[0].metadata["base_mask"], changed[0].metadata["base_mask"]))

    def test_invalid_roi_preserves_other_results(self):
        cfg = dict(self.alg.default_config(), use_rotation=False)
        results = self.alg.run_many(self.image, cfg, [(0, 0, 0, 0), self.boxes[0]])
        self.assertEqual(results[0].code, 2)
        self.assert_equal(results[1], self.alg.run(self.image, cfg, roi_bbox=self.boxes[0]))


if __name__ == "__main__":
    unittest.main()
