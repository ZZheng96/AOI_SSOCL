"""R3 精度：校准软限幅保序（合成数据，纯函数，不加载模型、不写存储）。
根因复现：旧 np.clip(q,-1,2) 把域外分数（> train 上沿）大量平局于 2.0，
TEST-07 上 shead raw AUROC 1.0 → cal 0.5、fused AUROC 0.935（修后 0.970）。"""
import unittest

import numpy as np
from sklearn.metrics import roc_auc_score

from algo.fusion.calibrate import CDFCalibrator, TplCalibrator, _soft_bound
from algo.fusion.fixed import fuse


def _p(msg):
    print(msg, flush=True)


class SoftBoundTests(unittest.TestCase):
    def test_range_monotone_and_identity_inside(self):
        _p("[r3 1/4] 软限幅：值域 [-1,2]、严格单调、[0,1] 内恒等")
        q = np.unique(np.concatenate([-np.geomspace(1e6, 1, 50), np.linspace(-1, 3, 400),
                                      np.geomspace(3, 1e6, 50)]))
        out = _soft_bound(q)
        self.assertTrue(np.all(out >= -1.0) and np.all(out <= 2.0))
        self.assertTrue(np.all(np.diff(out) > 0))
        inside = (q >= 0) & (q <= 1)
        np.testing.assert_allclose(out[inside], q[inside])
        # 边界连续（q=1 与 q=0 附近无跳变）
        self.assertAlmostEqual(float(_soft_bound(np.array([1.0 + 1e-9]))[0]), 1.0, places=6)
        self.assertAlmostEqual(float(_soft_bound(np.array([-1e-9]))[0]), 0.0, places=6)

    def test_cdf_out_of_domain_no_ties(self):
        _p("[r3 2/4] CDF：域外样本（远超 train 上沿）保序、不平局于 2.0")
        rng = np.random.default_rng(0)
        cal = CDFCalibrator(256).fit(rng.normal(0, 1, 20))
        far = np.array([5.0, 10.0, 20.0, 50.0, 100.0])     # 旧实现全部 = 2.0
        out = cal.transform(far)
        self.assertEqual(len(np.unique(out)), len(far))
        self.assertTrue(np.all(np.diff(out) > 0))
        self.assertTrue(np.all(out < 2.0) and np.all(out > 1.0))
        # 训练分布内行为不变：[0,1] 分位映射
        lo, hi = cal.edges[0], cal.edges[-1]
        inside = cal.transform(np.linspace(lo + 1e-6, hi, 30))
        self.assertTrue(np.all((inside >= 0.0) & (inside <= 1.0)))

    def test_tpl_soft_bound(self):
        _p("[r3 3/4] Tpl：diff/ref 软限幅保序，正常≈0 语义不变")
        cal = TplCalibrator().fit(np.full(10, 0.1))
        out = cal.transform(np.array([0.0, 0.05, 0.3, 1.0, 10.0]))
        self.assertAlmostEqual(float(out[0]), 0.0)
        self.assertAlmostEqual(float(out[1]), 0.5)
        self.assertTrue(np.all(np.diff(out) > 0))
        self.assertTrue(np.all(out <= 2.0))

    def test_fused_auroc_preserved_under_domain_shift(self):
        _p("[r3 4/4] 域差场景：test 全体超 train 上沿时 fused 仍保留 raw 排序")
        rng = np.random.default_rng(1)
        train = {"a": rng.normal(0, 1, 20), "b": rng.normal(0, 1, 20)}
        cals = {n: CDFCalibrator(256).fit(v) for n, v in train.items()}
        # 域差：test 正常与缺陷都 > train 上沿，缺陷更高（raw 可完全分离）
        tn = {"a": rng.uniform(4, 6, 15), "b": rng.uniform(4, 6, 15)}
        td = {"a": rng.uniform(7, 12, 15), "b": rng.uniform(7, 12, 15)}
        w = {"a": 0.5, "b": 0.5}
        fn = [fuse({n: float(cals[n].transform([tn[n][i]])[0]) for n in w}, w) for i in range(15)]
        fd = [fuse({n: float(cals[n].transform([td[n][i]])[0]) for n in w}, w) for i in range(15)]
        auc = roc_auc_score([0] * 15 + [1] * 15, fn + fd)
        self.assertEqual(auc, 1.0)   # 旧硬截断：全部 2.0 平局 → 0.5


if __name__ == "__main__":
    unittest.main(verbosity=2)
