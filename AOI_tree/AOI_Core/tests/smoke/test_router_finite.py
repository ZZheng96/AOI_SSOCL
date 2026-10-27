"""单缺陷路由诊断及检测输出有限值契约。"""
import json
import time
import unittest
from unittest.mock import Mock

import numpy as np
import torch

from algo.fusion.router import LinearRouter, train_router_on_matrix
from backend.engine import DetectionResult
from backend.pipeline.service import DetectionService


class RouterFiniteTest(unittest.TestCase):
    def test_single_positive_std_is_undefined(self):
        router, info = train_router_on_matrix(
            LinearRouter(2), [[0.9, 0.8]], [[0.1, 0.2], [0.2, 0.3]], epochs=2)
        self.assertIsNone(info["weight_std"])
        self.assertEqual(info["availability"]["weight_std"]["status"], "insufficient")
        self.assertTrue(info["availability"]["weight_std"]["reason"])
        self.assertTrue(torch.isfinite(router.W).all())
        json.dumps(info, allow_nan=False)

    def test_multiple_positive_preserves_sample_std(self):
        router, info = train_router_on_matrix(
            LinearRouter(2), [[0.9, 0.8], [0.7, 0.9]], [[0.1, 0.2]], epochs=2)
        with torch.no_grad():
            expected = router(torch.tensor([[0.9, 0.8], [0.7, 0.9]])).std(dim=0).mean().item()
        self.assertEqual(info["weight_std"], expected)
        self.assertEqual(info["availability"]["weight_std"]["status"], "available")
        json.dumps(info, allow_nan=False)

    def test_entropy_reg_resists_collapse(self):
        """CDEF6：λH 正则须提高权重熵（防坍缩），而不是鼓励坍缩。"""
        from algo.fusion.router import entropy_reg
        pos = [[0.9, 0.1, 0.5], [0.8, 0.2, 0.4], [0.95, 0.15, 0.6]]
        neg = [[0.1, 0.1, 0.4], [0.2, 0.15, 0.5], [0.15, 0.2, 0.45]]
        feats = torch.tensor(pos + neg)
        ent = {}
        for lam in (0.0, 5.0):
            router, _ = train_router_on_matrix(LinearRouter(3), pos, neg, epochs=200,
                                               lr=0.05, lam_entropy=lam, seed=0)
            with torch.no_grad():
                ent[lam] = float(entropy_reg(router(feats)))
        # 旧符号（+λH）在此设置下熵坍缩到 ≈0.004；修正后应保持接近 ln3。
        self.assertGreaterEqual(ent[5.0], ent[0.0] - 1e-6)
        self.assertGreater(ent[5.0], 0.9 * float(np.log(3)))

    def test_nonfinite_training_fails_closed(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                train_router_on_matrix(LinearRouter(2), [[value, 0.8]], [[0.1, 0.2]], epochs=2)

    def test_detection_rejects_invalid_before_persistence(self):
        service = DetectionService.__new__(DetectionService)
        service._persist = Mock(return_value=1)
        service._record_alarm = Mock(return_value={})
        result = DetectionResult(score=0.1234567890123456)
        out = service._finish_detect(result, np.zeros((2, 2, 3)), "sample.png", "test",
                                     time.perf_counter(), persist=False, with_heatmap=False)
        self.assertEqual(out["final_score"], result.score)
        self.assertEqual(out["decision"], result.decision)
        json.dumps(out, allow_nan=False)
        for invalid in (DetectionResult(score=float("nan")),
                        DetectionResult(extra={"candidate_router": {"train_info": {"weight_std": float("inf")}}})):
            with self.assertRaises(ValueError):
                service._finish_detect(invalid, np.zeros((2, 2, 3)), "sample.png", "test",
                                       time.perf_counter(), with_heatmap=False)
        service._persist.assert_not_called()
        self.assertEqual(service._record_alarm.call_count, 1)


if __name__ == "__main__":
    unittest.main()
