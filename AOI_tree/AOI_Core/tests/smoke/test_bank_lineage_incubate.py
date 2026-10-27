"""AC-028/029/032：库血缘、严格成簇、孵育纯 patch + 独立图计数 + 独立批准。"""
import unittest

import numpy as np
import torch

from algo.ssocl.banks import NormalBank, DefectBank
from algo.ssocl.incubate import HeadManager

D, G = 16, 6


def _tile(seed, center=None, noise=0.05):
    rng = np.random.default_rng(seed)
    if center is None:
        center = rng.normal(size=D)
    x = center[None, :, None, None] + noise * rng.normal(size=(1, D, G, G))
    return x.astype(np.float32)


class TestNormalBankCluster(unittest.TestCase):
    def _bank(self):
        nb = NormalBank(cluster_k=3, sim_thresh=-1.0)
        nb.seed_core(np.random.default_rng(0).normal(size=(2, D, G, G)).astype(np.float32))
        return nb

    def test_dissimilar_candidates_stay_pending(self):
        nb = self._bank()
        rs = [nb.reflow_candidate(_tile(100 + i)) for i in range(5)]
        self.assertEqual(rs, ["pending"] * 5)        # 互不相似：不同簇，均不成簇
        self.assertEqual(len(nb.cluster_hits), 5)

    def test_similar_candidates_accept_after_k(self):
        nb = self._bank()
        c = np.random.default_rng(7).normal(size=D)
        rs = [nb.reflow_candidate(_tile(200 + i, center=c)) for i in range(3)]
        self.assertEqual(rs, ["pending", "pending", "accepted"])
        self.assertEqual(nb.cluster_hits, {})

    def test_window_expiry_breaks_continuity(self):
        nb = self._bank()
        nb.cluster_window = 2
        c = np.random.default_rng(8).normal(size=D)
        nb.reflow_candidate(_tile(300, center=c))
        for i in range(3):                            # 插入 3 个无关样本 → 超窗
            nb.reflow_candidate(_tile(400 + i))
        self.assertEqual(nb.reflow_candidate(_tile(301, center=c)), "pending")
        self.assertEqual(nb.reflow_candidate(_tile(302, center=c)), "pending")

    def test_lineage_and_rollback(self):
        nb = self._bank()
        nb.snapshot()
        self.assertTrue(nb.add_ext(_tile(1), source="reflow", min_feat=1, image_id="a.png"))
        lin = nb.lineage()
        self.assertTrue(lin["memory_not_weights"])
        self.assertEqual(lin["core"]["source"], "train_good_anchor")
        e = lin["ext"][-1]
        self.assertEqual((e["source"], e["image_id"], e["v"]), ("reflow", "a.png", 1))
        self.assertEqual(len(e["hash"]), 16)
        nb.rollback()
        self.assertEqual(nb.lineage()["ext"], [])

    def test_legacy_object_without_meta(self):
        nb = self._bank()
        nb.add_ext(_tile(2), min_feat=1)
        del nb.ext_meta                               # 模拟旧 pickle
        lin = nb.lineage()
        self.assertTrue(lin["ext"][0]["legacy"])
        nb.add_ext(_tile(3), min_feat=1)              # 旧对象继续可写
        self.assertEqual(len(nb.lineage()["ext"]), 2)


class TestDefectBankPool(unittest.TestCase):
    def test_whole_image_not_in_pool_and_boxes_count_by_image(self):
        db = DefectBank()
        db.add(_tile(1), image_id="whole.png")
        db.add_box(_tile(2), [0.0, 0.0, 0.5, 0.5], image_id="img1.png")
        db.add_box(_tile(2), [0.5, 0.5, 1.0, 1.0], image_id="img1.png")
        db.add_patch_feats(np.random.default_rng(3).normal(size=(8, D)), image_id="img2.png")
        feats, ids = db.incubation_pool()
        self.assertEqual(len(feats), 3)
        self.assertEqual(sorted(set(ids)), ["img1.png", "img2.png"])
        lin = db.lineage()
        self.assertEqual(lin["n_pure_images"], 2)
        self.assertEqual([m["op"] for m in lin["samples"]],
                         ["add", "add_box", "add_box", "add_patch"])
        self.assertEqual(lin["samples"][1]["box"], [0.0, 0.0, 0.5, 0.5])

    def test_legacy_samples_excluded(self):
        db = DefectBank()
        db.add_box(_tile(2), [0.0, 0.0, 0.5, 0.5], image_id="x")
        db.sample_meta = []                           # 旧对象：血缘缺失
        self.assertEqual(db.incubation_pool(), ([], []))
        self.assertTrue(db.lineage()["samples"][0]["legacy"])


class TestPostFusionRoles(unittest.TestCase):
    """AC-033：核心槽职责字段 + 融合后拦截通道状态，且不进融合。"""

    def test_roles_and_runtime(self):
        from types import SimpleNamespace
        from algo.fusion import cde_report as cr
        db = DefectBank()
        db.add_box(_tile(3), [0.0, 0.0, 0.5, 0.5], image_id="a")
        db.add(_tile(4), image_id="b")                      # 整图 add 不算纯 patch
        pipe = SimpleNamespace(slots={"disc": 1, "sem": 1, "inp": 1}, open_det=None,
                               calibrators={}, slot_sanity={}, weights={"disc": 1.0},
                               handler=SimpleNamespace(defect_bank=db),
                               head_mgr=HeadManager({"min_samples": 4}))
        h = cr.slot_health(pipe)
        for n in ("disc", "sem", "inp"):
            self.assertTrue({"role", "train", "update", "not", "localization",
                             "domain_sensitivity"} <= set(h["slots"][n]))
        pf = h["post_fusion"]
        self.assertFalse(pf["defect_knn"]["in_fusion"])
        self.assertTrue(pf["defect_knn"]["active"])
        self.assertEqual(pf["defect_knn"]["n_samples"], 2)
        self.assertEqual(pf["incubate"]["n_pure_images"], 1)
        self.assertFalse(pf["incubate"]["trained"])
        self.assertIn("不替代 disc", pf["incubate"]["not"])
        self.assertNotIn("incubate", h["slots"])
        empty = cr.post_fusion_health(SimpleNamespace())
        self.assertFalse(empty["defect_knn"]["active"])
        self.assertFalse(empty["incubate"]["enabled"])


class TestIncubateApproval(unittest.TestCase):
    def _bank(self, n_img, center, noise=0.05):
        db = DefectBank()
        for i in range(n_img):
            for b in ([0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0]):   # 同图两框
                db.add_box(_tile(1000 + i, center=center, noise=noise), b,
                           image_id=f"img{i}.png")
        return db

    def setUp(self):
        rng = np.random.default_rng(42)
        self.core = rng.normal(size=(4, D, G, G)).astype(np.float32)   # 4D 锚定负样本
        self.center = rng.normal(size=D) * 3

    def test_counts_images_not_boxes(self):
        hm = HeadManager({"min_samples": 4})
        self.assertIsNone(hm.maybe_incubate(self._bank(3, self.center), self.core))
        self.assertEqual(hm.status(self._bank(3, self.center))["n_pure_images"], 3)

    def test_approved_with_4d_core(self):
        hm = HeadManager({"min_samples": 4, "holdout_images": 2})
        h = hm.maybe_incubate(self._bank(4, self.center), self.core)
        self.assertIsNotNone(h)
        rec = hm.log[-1]
        self.assertTrue(rec["approved"])
        self.assertEqual(rec["reason"], "holdout_pass")
        self.assertEqual(rec["n_images"], 4)
        self.assertGreater(rec["sep_new"], 0.2)
        self.assertGreater(h.score(_tile(5, center=self.center)), h.score(self.core[:1]))

    def test_rejected_keeps_old_head_and_waits(self):
        hm = HeadManager({"min_samples": 4, "holdout_images": 2,
                          "approve_min_sep": 1.01})       # 不可达门槛 → 必拒
        self.assertIsNone(hm.maybe_incubate(self._bank(4, self.center), self.core))
        self.assertEqual(hm.log[-1]["reason"], "below_min_sep")
        n_log = len(hm.log)
        hm.maybe_incubate(self._bank(5, self.center), self.core)   # 仅 +1 图：不重试
        self.assertEqual(len(hm.log), n_log)
        hm.maybe_incubate(self._bank(6, self.center), self.core)   # +2 图：重试
        self.assertEqual(len(hm.log), n_log + 1)
        self.assertIsNone(hm.head)

    def test_worse_than_current_rejected(self):
        hm = HeadManager({"min_samples": 4, "holdout_images": 2})
        old = hm.maybe_incubate(self._bank(4, self.center), self.core)
        hm.cfg["approve_min_sep"] = -1.0
        noisy = self._bank(6, self.center, noise=0.05)
        # 新训练集换成纯噪声正样本 → 新头 holdout 不如旧头
        orig = noisy.incubation_pool

        def pool():
            feats, ids = orig()
            hold = set(list(dict.fromkeys(ids))[-2:])
            rng = np.random.default_rng(9)
            feats = [f if i in hold else torch.nn.functional.normalize(
                torch.as_tensor(rng.normal(size=tuple(f.shape)), dtype=torch.float32),
                dim=1).half() for f, i in zip(feats, ids)]
            return feats, ids
        noisy.incubation_pool = pool
        hm.maybe_incubate(noisy, self.core)
        self.assertEqual(hm.log[-1]["reason"], "worse_than_current")
        self.assertIs(hm.head, old)


if __name__ == "__main__":
    unittest.main()
