"""R4 sem 方向 / shead 置零根因回归（合成特征，CPU，不加载 DINO、不写存储）。
结论：sem 打分 = 1 - max cos（距离，越大越异常）、CDF 校准同向保序，代码无符号反转；
gold_finger 上 sem AUROC 0.30 / domain 下 sem·shead sanity ≤0.5 均来自数据：
init_defect 的 *_repaired 配对图在正常库内（缺陷图≈库成员 → 距离小），test 正常批离库更远。"""
import unittest

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from algo.eval.offline import resolve_isolation
from algo.fusion.calibrate import CDFCalibrator
from algo.slots.sem import SemSlot


def _p(msg):
    print(msg, flush=True)


def _tiles(base, noise, g, n=1, G=4):
    """(n,D,G,G) tile 特征：base 方向 + 高斯扰动。"""
    D = base.shape[0]
    f = base.view(1, D, 1, 1) + noise * torch.randn(n, D, G, G, generator=g)
    return f


class SemDirectionTests(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(0)
        self.g = g
        self.D = 32
        self.base = F.normalize(torch.randn(self.D, generator=g), dim=0)
        train = [_tiles(self.base, 0.05, g) for _ in range(8)]
        self.slot = SemSlot({"coreset_size": 16, "seed": 0}, device="cpu").fit(
            {"train_tile_feats": train})

    def _score(self, f):
        ts, _ = self.slot.score_tiles(f)
        return float(np.asarray(ts).max())

    def test_far_scores_higher(self):
        _p("[r4 1/4] sem：离正常库越远分越高（距离语义，非相似度）")
        near = [self._score(_tiles(self.base, 0.05, self.g)) for _ in range(10)]
        far = []
        for _ in range(10):
            f = _tiles(self.base, 0.05, self.g)
            f[:, :, 1, 1] = torch.randn(1, self.D, generator=self.g) * 2.0   # 局部异常 patch
            far.append(self._score(f))
        auc = roc_auc_score([0] * 10 + [1] * 10, near + far)
        self.assertGreater(auc, 0.95)

    def test_cdf_preserves_direction(self):
        _p("[r4 2/4] CDF 校准单调递增：raw 排序 = cal 排序（不翻转 sem 方向）")
        rng = np.random.default_rng(0)
        cal = CDFCalibrator(256).fit(rng.normal(0.4, 0.05, 20))
        x = np.linspace(0.0, 1.5, 200)
        self.assertTrue(np.all(np.diff(cal.transform(x)) >= 0))

    def test_twin_confound_reverses_auroc(self):
        _p("[r4 3/4] 配对混淆：缺陷图≈库内 repaired 孪生 → 分低于离库正常（数据致 AUROC<0.5）")
        g = self.g
        other = F.normalize(self.base + 1.0 * F.normalize(torch.randn(self.D, generator=g), dim=0), dim=0)
        # test 正常：另一采集批（离库远）；test 缺陷：库内批 + 小缺陷 patch
        normals = [self._score(_tiles(other, 0.01, g)) for _ in range(10)]
        defects = []
        for _ in range(10):
            f = _tiles(self.base, 0.01, g)
            f[:, :, 0, 0] = f[:, :, 0, 0] + 0.03 * torch.randn(1, self.D, generator=g)
            defects.append(self._score(f))
        self.assertLess(roc_auc_score([0] * 10 + [1] * 10, normals + defects), 0.5)


class SheadSanityTests(unittest.TestCase):
    def test_domain_isolation_holdout(self):
        _p("[r4 4/4] domain 36 张正常 → holdout 7 校准 / 29 拟合（sanity 用留出正常，诚实非 bug）")
        iso, cal_idx, fit_idx, info = resolve_isolation({"isolation": "auto"}, 36, 42)
        self.assertEqual(iso, "holdout")
        self.assertEqual((len(cal_idx), len(fit_idx)), (7, 29))
        self.assertFalse(set(cal_idx) & set(fit_idx))
        iso_b, cal_b, fit_b, _ = resolve_isolation({"isolation": "auto"}, 14, 42)
        self.assertEqual(iso_b, "self")       # baseline 14 张回退 self（sanity 含自匹配，偏乐观）


if __name__ == "__main__":
    unittest.main(verbosity=2)
