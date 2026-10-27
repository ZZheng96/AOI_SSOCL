"""AOI_feature 与 Core Trad 的少量逐图等价契约测试。

只使用确定性内存图，不读取数据集、不修改生产配置或快照。
默认使用 32x32，使 selfsim 不触发随机抽样；随机性另由边界测试明确记录。
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AOI_feature.feature_set import FeatureSet
from AOI_feature.presets import trunk_preset
from algo.features import registry as freg
from algo.vendor.traditional import TraditionalFeatureExtractor


class FeatureParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fs = FeatureSet(trunk_preset())
        cls.core = TraditionalFeatureExtractor()
        cls.images = [
            np.full((32, 32, 3), 127, dtype=np.uint8),
            np.indices((32, 32), dtype=np.uint16)[0].astype(np.uint8)[..., None]
            .repeat(3, axis=2),
            np.random.default_rng(20261014).integers(
                0, 256, (32, 32, 3), dtype=np.uint8),
        ]

    def test_registry_and_experiment_dimensions(self):
        print("[parity 1/3] 检查组顺序、切片和总维度", flush=True)
        meta = self.fs.probe()
        self.assertEqual([m["key"] for m in meta], list(freg.GROUP_KEYS))
        self.assertEqual([m["dim"] for m in meta],
                         [g["dim"] for g in freg.TRAD_GROUPS])
        self.assertEqual(self.fs.dim, freg.TRAD_DIM)
        self.assertEqual(self.fs.dim, 200)

    def test_representative_images_finite_and_equal(self):
        print("[parity 2/3] 3 张内存代表图逐图比较 Core/AOI_feature", flush=True)
        for i, image in enumerate(self.images):
            with self.subTest(image=i):
                expected = self.core.extract(image, rng=np.random.default_rng(42))
                actual = self.fs.extract_one(image)
                self.assertEqual(expected.shape, (200,))
                self.assertEqual(actual.shape, (200,))
                self.assertTrue(np.isfinite(expected).all())
                self.assertTrue(np.isfinite(actual).all())
                np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    def test_random_sampling_uses_explicit_shared_rng(self):
        print("[parity 3/4] 大图 selfsim 使用共享 default_rng(42)", flush=True)
        image = np.random.default_rng(7).integers(0, 256, (256, 256, 3), dtype=np.uint8)
        actual = self.fs.extract_one(image, rng=np.random.default_rng(42))
        expected = self.core.extract(image, rng=np.random.default_rng(42))
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    def test_groupknn_fit_matches_production_float32_bank(self):
        print("[parity 4/4] GroupKNN fit 与生产 float32 口径逐位一致", flush=True)
        """协议 B：候选 fit 与生产 TradSlot.fit 使用同样的 float32 口径。"""
        from algo.slots.trad import TradSlot
        from AOI_feature.scoring import GroupKNN

        train = np.stack(
            [self.fs.extract_one(img, rng=np.random.default_rng(42))
             for img in self.images]
        ).astype(np.float32)

        slot = TradSlot({})
        slot.fit({"train_imgs": self.images})
        cand = GroupKNN().fit(train)

        np.testing.assert_array_equal(
            cand.bank,
            slot.bank_w,
        )
        np.testing.assert_array_equal(
            cand.mean,
            slot.norm_w[0],
        )
        np.testing.assert_array_equal(
            cand.std,
            slot.norm_w[1],
        )


if __name__ == "__main__":
    unittest.main()
