"""round5 L3 配对模板匹配（tpl pair_mode）：合成数据，桩 backbone，不加载模型/数据集。"""
import os
import tempfile
import unittest

import cv2
import numpy as np
import torch

from algo.slots.tpl import TplSlot, resolve_pair_template


class _StubBackbone:
    """确定性特征：图像灰度下采样到 G×G，复制成 384 通道（局部差异 → 局部特征差）。"""
    grid = 8

    def extract_tiles(self, imgs):
        out = []
        for im in imgs:
            g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if im.ndim == 3 else im
            s = cv2.resize(g, (self.grid, self.grid), interpolation=cv2.INTER_AREA)
            t = torch.tensor(s, dtype=torch.float32) / 255.0
            out.append(t.unsqueeze(0).repeat(384, 1, 1))
        return torch.stack(out)


def _board(seed):
    rng = np.random.default_rng(seed)
    img = np.full((112, 112, 3), 90, np.uint8)
    for _ in range(12):
        x, y = rng.integers(0, 100, 2)
        img[y:y + 10, x:x + 6] = rng.integers(150, 250)
    return img


def _defect(img):
    d = img.copy()
    d[40:60, 40:60] = 255
    return d


def _slot(**kw):
    cfg = {"pix_weight": 0.0, "feat_weight": 1.0, "topk_ratio": 0.1}
    cfg.update(kw)
    return TplSlot(cfg, _StubBackbone(), device="cpu")


class PairModeTest(unittest.TestCase):
    def setUp(self):
        self.lib = [_board(i) for i in range(4)]
        self.good = _board(10)
        self.bad = _defect(self.good)

    def test_default_off_unchanged(self):
        a = _slot().fit({"templates": self.lib})
        b = _slot().fit({"templates": self.lib})
        self.assertFalse(a.pair_mode)
        r0 = a.score_image(self.bad)
        r1 = b.score_image(self.bad, pair_template=self.good)   # 关闭时忽略 pair_template
        self.assertEqual(r0[0], r1[0])
        np.testing.assert_array_equal(r0[2][0], r1[2][0])

    def test_pair_defect_above_its_repaired(self):
        s = _slot(pair_mode=True).fit({"templates": self.lib})
        sd = s.score_image(self.bad, pair_template=self.good)[0]
        sg = s.score_image(self.good, pair_template=self.good)[0]
        self.assertGreater(sd, sg)
        self.assertLess(sg, 1e-6)
        # 另一张修复图配对（口径C）也低于缺陷图
        so = s.score_image(_board(11), pair_template=self.good)[0]
        self.assertGreater(so, 0.0)

    def test_resolve_names(self):
        with tempfile.TemporaryDirectory() as d:
            d2 = os.path.join(d, "tpl")
            os.makedirs(d2)
            x = os.path.join(d, "1308.jpg")
            open(x, "wb").close()
            self.assertIsNone(resolve_pair_template(x))
            self.assertIsNone(resolve_pair_template(x, [d2]))
            t2 = os.path.join(d2, "1308_repaired.png")
            open(t2, "wb").close()
            self.assertEqual(resolve_pair_template(x, [d2]), t2)
            t1 = os.path.join(d, "1308_repaired.jpg")
            open(t1, "wb").close()
            self.assertEqual(resolve_pair_template(x, [d2]), t1)          # 同目录优先
            self.assertIsNone(resolve_pair_template(t1, [d2]))            # 正常图本身 → None
            self.assertIsNone(resolve_pair_template(os.path.join(d, "13080.jpg"), [d2]))

    def test_missing_pair_falls_back(self):
        a = _slot(pair_mode=True).fit({"templates": self.lib})
        b = _slot().fit({"templates": self.lib})
        self.assertEqual(a.score_image(self.bad, pair_template=None)[0],
                         b.score_image(self.bad)[0])
        # Pipeline 层：无配对 → _pair_kw 为空（走原逻辑）
        from algo.eval.offline import Pipeline
        p = Pipeline.__new__(Pipeline)
        p.cfg = {"slots": {"tpl": {"pair_mode": True, "pair_auto_resolve": True,
                                   "pair_dirs": []}}}
        self.assertIsNone(p._resolve_pair(os.path.join(tempfile.gettempdir(),
                                                       "no_such_x.jpg")))
        p._pair_ctx = None
        self.assertEqual(p._pair_kw(a), {})
        p._pair_ctx = "t.png"
        self.assertEqual(p._pair_kw(a), {"pair_template": "t.png"})
        self.assertEqual(p._pair_kw(b), {})        # pair_mode 关闭的槽位不透传


if __name__ == "__main__":
    unittest.main()
