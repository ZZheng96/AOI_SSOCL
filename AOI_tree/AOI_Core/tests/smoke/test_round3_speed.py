"""round3_speed（HARD-01）：推理路径提速的等价性守护。

- blob 无损改写（σ 去重/cv2.absdiff+max/精确 top-k 预筛/原地归一）与旧实现逐位一致；
- GPU DoG 有损开关默认关；开启时与 CPU 相对差 < 1e-5；
- Pipeline CPU 线程池 worker 自适应在 [2, 4]。
"""
import os
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from algo.slots.blob import BlobSlot


def _ref_scores(tiles, topk=10):
    """R3 之前的参考实现（逐字等价）。"""
    out = []
    for t in tiles:
        g = cv2.cvtColor(t, cv2.COLOR_RGB2GRAY) if t.ndim == 3 else t
        g = g.astype(np.float32) / 255.0
        b1 = cv2.GaussianBlur(g, (0, 0), 1.0)
        b2 = cv2.GaussianBlur(g, (0, 0), 2.0)
        b2b = cv2.GaussianBlur(g, (0, 0), 2.0)
        b4 = cv2.GaussianBlur(g, (0, 0), 4.0)
        r = np.maximum(np.abs(b1 - b2), np.abs(b2b - b4))
        k = min(topk, r.size)
        out.append(float(np.partition(r.ravel(), -k)[-k:].mean()))
    return np.asarray(out)


def _tiles():
    rng = np.random.default_rng(0)
    ts = [rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
          for h, w in [(512, 512), (1200, 1600), (7, 9), (1, 1)]]
    # 对抗样例：top-k 全落在步长子集之外 / 大量并列值
    a = np.full((1024, 1024, 3), 128, np.uint8)
    for i in range(12):
        a[3 + 37 * i, 5 + 41 * i] = 255
    ts.append(a)
    ts.append(np.zeros((800, 800), np.uint8))
    return ts


class TestBlobLossless(unittest.TestCase):
    def test_bitwise_equal_to_reference(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AOI_BLOB_GPU", None)
            s = BlobSlot({}, "cpu")
            ts = _tiles()
            got, _ = s.score_tiles(tile_imgs=ts)
            np.testing.assert_array_equal(got, _ref_scores(ts))

    def test_topk_mean_exact(self):
        rng = np.random.default_rng(1)
        for n in (5, 100, 4096, 300_001):
            r = rng.random(n, dtype=np.float32)
            for k in {1, min(n, 10), min(n, 50)}:
                ref = float(np.partition(r, -k)[-k:].mean())
                self.assertEqual(BlobSlot._topk_mean(r, k), ref)

    def test_gpu_off_by_default(self):
        os.environ.pop("AOI_BLOB_GPU", None)
        self.assertIsNone(BlobSlot({}, "cuda")._dev)
        with patch.dict(os.environ, {"AOI_BLOB_GPU": "0"}):
            self.assertIsNone(BlobSlot({"gpu": True}, "cuda")._dev)


@unittest.skipUnless(torch.cuda.is_available(), "no cuda")
class TestBlobGpuSwitch(unittest.TestCase):
    def test_gpu_close_to_cpu(self):
        os.environ.pop("AOI_BLOB_GPU", None)
        cpu = BlobSlot({}, "cpu")
        gpu = BlobSlot({"gpu": True, "gpu_min_px": 1}, "cuda")
        self.assertIsNotNone(gpu._dev)
        ts = _tiles()[:2] + [_tiles()[4]]
        a, _ = cpu.score_tiles(tile_imgs=ts)
        b, _ = gpu.score_tiles(tile_imgs=ts)
        np.testing.assert_allclose(b, a, rtol=1e-5, atol=1e-7)

    def test_small_image_falls_back_cpu(self):
        gpu = BlobSlot({"gpu": True}, "cuda")
        self.assertFalse(gpu._gpu_ok(512, 512, 1.0))   # < gpu_min_px（1MP）
        self.assertFalse(gpu._gpu_ok(20, 4000, 1.0))   # 边长不足 reflect pad


class TestCpuExecWorkers(unittest.TestCase):
    def test_workers_adaptive(self):
        from algo.eval.offline import Pipeline
        for n in (None, 1, 4, 8, 64):
            with patch("os.cpu_count", return_value=n):
                p = Pipeline({"tiling": {}}, None, [])
                w = p._cpu_exec._max_workers
                p._cpu_exec.shutdown(wait=False)
                self.assertTrue(2 <= w <= 4, (n, w))


if __name__ == "__main__":
    unittest.main()
