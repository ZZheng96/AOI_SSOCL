"""按缺陷类型孵育：多类型分别孵育 / 样本不足不孵育 / 旧状态兼容 / boost 只用通用头。"""
import pickle
import unittest

import numpy as np

from algo.ssocl.banks import DefectBank
from algo.ssocl.incubate import HeadManager

D, G = 16, 6
BOXES = ([0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0])


def _tile(seed, center, noise=0.05):
    rng = np.random.default_rng(seed)
    x = center[None, :, None, None] + noise * rng.normal(size=(1, D, G, G))
    return x.astype(np.float32)


class TestPerTypeIncubate(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.core = rng.normal(size=(4, D, G, G)).astype(np.float32)
        self.c_a = rng.normal(size=D) * 3
        self.c_b = rng.normal(size=D) * 3

    def _fill(self, db, n_img, center, dtype, prefix, seed0):
        for i in range(n_img):
            for b in BOXES:
                db.add_box(_tile(seed0 + i, center), b, image_id=f"{prefix}{i}.png",
                           defect_type=dtype)

    def _cfg(self, **kw):
        return {"min_samples": 4, "holdout_images": 2, **kw}

    def test_multi_type_heads_trained_separately(self):
        print("[+] 多类型分别孵育：scratch/dent 各自成头，通用头不变", flush=True)
        db = DefectBank()
        self._fill(db, 4, self.c_a, "scratch", "a", 100)
        self._fill(db, 4, self.c_b, "dent", "b", 200)
        hm = HeadManager(self._cfg())
        h = hm.maybe_incubate(db, self.core)
        self.assertIsNotNone(h)
        self.assertIs(h, hm.head)
        self.assertEqual(set(hm.heads), {"scratch", "dent"})
        self.assertIsNot(hm.heads["scratch"], hm.heads["dent"])
        # 类型头日志带 defect_type；通用头日志不带
        typed_logs = [e for e in hm.log if "defect_type" in e]
        self.assertEqual({e["defect_type"] for e in typed_logs}, {"scratch", "dent"})
        self.assertTrue(all(e["approved"] for e in typed_logs))
        st = hm.status(db)
        self.assertNotIn("defect_type", st["last_decision"])
        self.assertTrue(st["per_type"]["scratch"]["trained"])
        self.assertEqual(st["per_type"]["dent"]["n_pure_images"], 4)
        # 类型头专精：scratch 头对 scratch 样本打分高于 dent 样本
        ts_a = hm.type_scores(_tile(999, self.c_a))
        ts_b = hm.type_scores(_tile(998, self.c_b))
        self.assertGreater(ts_a["scratch"], ts_b["scratch"])
        self.assertGreater(ts_b["dent"], ts_a["dent"])

    def test_insufficient_type_not_incubated(self):
        print("[+] 类型样本不足不孵育；无类型归 _any 只进通用头", flush=True)
        db = DefectBank()
        self._fill(db, 4, self.c_a, "scratch", "a", 100)
        self._fill(db, 2, self.c_b, "dent", "b", 200)        # dent 仅 2 张 < 4
        self._fill(db, 3, self.c_b, None, "u", 300)          # 无类型
        hm = HeadManager(self._cfg())
        self.assertIsNotNone(hm.maybe_incubate(db, self.core))
        self.assertEqual(set(hm.heads), {"scratch"})
        self.assertNotIn("_any", hm.heads)
        st = hm.status(db)
        self.assertFalse(st["per_type"]["dent"]["trained"])
        self.assertEqual(st["n_pure_images"], 9)
        _, _, types = db.incubation_pool_typed()
        self.assertIn("_any", types)

    def test_no_types_keeps_single_head_behavior(self):
        print("[+] 无类型样本：行为与原单头一致（heads 为空）", flush=True)
        db = DefectBank()
        self._fill(db, 3, self.c_a, None, "a", 100)
        hm = HeadManager(self._cfg())
        self.assertIsNone(hm.maybe_incubate(db, self.core))   # 通用池不足 → None
        self._fill(db, 2, self.c_a, None, "x", 400)
        self.assertIsNotNone(hm.maybe_incubate(db, self.core))
        self.assertEqual(hm.heads, {})
        self.assertEqual(hm.type_scores(_tile(1, self.c_a)), {})

    def test_boost_uses_only_generic_head(self):
        print("[+] boost 只用通用头，类型头不影响在线打分", flush=True)
        db = DefectBank()
        self._fill(db, 4, self.c_a, "scratch", "a", 100)
        hm = HeadManager(self._cfg())
        hm.maybe_incubate(db, self.core)
        x = _tile(7, self.c_a)
        b0 = hm.boost(x)
        self.assertEqual(b0, hm.head.score(x))
        hm.heads.clear()
        self.assertEqual(hm.boost(x), b0)

    def test_per_type_disabled(self):
        db = DefectBank()
        self._fill(db, 4, self.c_a, "scratch", "a", 100)
        hm = HeadManager(self._cfg(per_type=False))
        hm.maybe_incubate(db, self.core)
        self.assertEqual(hm.heads, {})

    def test_legacy_pickle_compat(self):
        print("[+] 旧单头 pickle（无 heads/type_state）可加载并继续按类型孵育", flush=True)
        db = DefectBank()
        self._fill(db, 4, self.c_a, None, "a", 100)
        hm = HeadManager(self._cfg())
        hm.maybe_incubate(db, self.core)
        old_head_n = hm.head.n_pos
        for k in ("heads", "type_state", "last_sep"):
            del hm.__dict__[k]                               # 模拟旧格式
        hm2 = pickle.loads(pickle.dumps(hm))
        self.assertEqual(hm2.heads, {})
        self.assertEqual(hm2.type_state, {})
        self.assertIsNone(hm2.last_sep)
        self.assertEqual(hm2.head.n_pos, old_head_n)
        self.assertEqual(hm2.status(db)["per_type"], {})
        self._fill(db, 4, self.c_b, "dent", "b", 200)
        hm2.maybe_incubate(db, self.core)
        self.assertIn("dent", hm2.heads)

    def test_legacy_bank_without_typed_pool(self):
        db = DefectBank()
        self._fill(db, 4, self.c_a, "scratch", "a", 100)
        db.incubation_pool_typed = None                      # 旧库对象无该方法
        hm = HeadManager(self._cfg())
        self.assertIsNotNone(hm.maybe_incubate(db, self.core))
        self.assertEqual(hm.heads, {})


if __name__ == "__main__":
    unittest.main()
