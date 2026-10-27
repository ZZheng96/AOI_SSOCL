"""AC-030/031：四类反馈分路、无反馈默认不回流、缺陷入库锚定 FPR 门控、init_defect 预置。"""
import types
import unittest

import numpy as np
import torch

from algo.ssocl.banks import DefectBank
from algo.ssocl.feedback import FeedbackHandler
from algo.eval.offline import _masked_defect_patches

D, G = 16, 6


class _Gate:
    """锚定门控 mock：按预设返回 drift 与 anchor_fpr。"""
    def __init__(self, drift=0.0, fpr=(0.0, 0.0), anchors=("a.png",)):
        self.anchor_paths = list(anchors)
        self.base_mean, self.tol = 0.0, 1.0
        self.drift, self.fpr, self.audit = drift, fpr, []

    def check(self, p, tag):
        self.audit.append({"tag": tag, "anchor_fpr": {"before": self.fpr[0],
                                                      "after": self.fpr[1]}})
        return self.drift <= self.tol, self.drift, self.base_mean + self.drift


def _handler(gate=None, **kw):
    h = object.__new__(FeedbackHandler)               # 不跑 __init__：同时覆盖旧 pickle getattr 兜底
    h.p = types.SimpleNamespace(
        decider=types.SimpleNamespace(tau_high=0.8, tau_gray=0.5),
        active_enqueue=lambda rec, item: None, _ablate_reflow=False)
    h.gate = gate or _Gate()
    h.defect_bank = DefectBank()
    h.stats = {"gate_reject": 0, "rollback": 0}
    h.rollback_log, h.high_conf = [], []
    h._reflow = lambda item, src, labeled=False: {"action": "reflow_accepted", "src": src}
    for k, v in kw.items():
        setattr(h, k, v)
    return h


def _patches(seed, k=8):
    return np.random.default_rng(seed).normal(size=(k, D)).astype(np.float32)


class TestRoutes(unittest.TestCase):
    def test_route_of(self):
        r = FeedbackHandler.route_of
        self.assertEqual(r("correct", None), "unlabeled")
        self.assertEqual(r("correct", 1), "defect")
        self.assertEqual(r("wrong", 1), "defect")
        self.assertEqual(r("correct", 0), "correct_normal")
        self.assertEqual(r("wrong", 0), "false_positive")
        self.assertEqual(r("review", 0), "review_normal")


class TestUnlabeled(unittest.TestCase):
    item = {"feats": np.zeros((1, D, G, G), np.float32)}

    def _rec(self, fused, op=0.1):
        return {"path": "x.png", "fused": fused, "open_score": op}

    def test_default_no_reflow(self):
        h = _handler()
        self.assertEqual(h._unlabeled(self.item, self._rec(0.05))["action"], "low_value_skip")
        self.assertEqual(h._unlabeled(self.item, self._rec(0.9))["action"], "high_conf_cache")
        self.assertEqual(h._unlabeled(self.item, self._rec(0.6))["action"], "active_queue")

    def test_enabled_gates(self):
        h = _handler(unlabeled_reflow=True, ua_max_fused=0.2, ua_max_open=0.5)
        self.assertEqual(h._unlabeled(self.item, self._rec(0.3))["reason"],
                         "not_high_conf_normal")
        self.assertEqual(h._unlabeled(self.item, self._rec(0.1, op=0.9))["reason"],
                         "domain_unverified")
        self.assertEqual(h._unlabeled(self.item, self._rec(0.1, op=None))["reason"],
                         "domain_unverified")
        out = h._unlabeled(self.item, self._rec(0.1))
        self.assertEqual((out["action"], out["src"]), ("reflow_accepted", "unlabeled_auto"))
        self.assertEqual(h.stats["unlabeled_reflow"], 1)


class TestDefectGate(unittest.TestCase):
    def _add(self, h, seed):
        h.defect_bank.snapshot("t")
        h.defect_bank.add_patch_feats(_patches(seed), "fb", image_id=f"{seed}.png")
        return h._defect_bank_gate("fb")

    def test_fpr_up_rolls_back(self):
        h = _handler(_Gate(fpr=(0.0, 0.05)))
        self.assertFalse(self._add(h, 1))
        self.assertEqual(len(h.defect_bank.samples), 0)
        self.assertEqual(h.rollback_log[-1]["reason"], "anchor_fpr_up")
        self.assertEqual(h.gate.audit[-1]["decision"], "rolled_back")

    def test_tolerance_and_drift(self):
        h = _handler(_Gate(fpr=(0.0, 0.05)), defect_fpr_tol=0.1)
        self.assertTrue(self._add(h, 2))
        self.assertEqual(len(h.defect_bank.samples), 1)
        h.gate.drift = 2.0                             # 均值漂移超 tol
        self.assertFalse(self._add(h, 3))
        self.assertEqual(h.rollback_log[-1]["reason"], "regress_gate")
        self.assertEqual(len(h.defect_bank.samples), 1)

    def test_scope_similar_only(self):
        h = _handler()
        self._add(h, 4)
        s = h._defect_scope(True)
        self.assertEqual((s["scope"], s["n_pure_images"], s["defect_gate"]),
                         ("similar_only", 1, "accepted"))
        h.defect_scope_min = 1
        self.assertEqual(h._defect_scope(False)["scope"], "incubation_ready")
        self.assertEqual(h._defect_scope(False)["defect_gate"], "rolled_back")


class TestSeedInitDefect(unittest.TestCase):
    def test_accept_rollback_skip(self):
        h = _handler()
        self.assertEqual(h.seed_init_defect([])["action"], "init_defect_seed_skip")
        self.assertEqual(h.seed_init_defect([("a", _patches(0, k=2))])["action"],
                         "init_defect_seed_skip")         # K<4 全部被拒
        self.assertEqual(len(h.defect_bank.samples), 0)
        r = h.seed_init_defect([("a", _patches(1)), ("b", _patches(2))])
        self.assertEqual((r["action"], r["n"]), ("init_defect_seed_accepted", 2))
        h2 = _handler(_Gate(fpr=(0.0, 0.1)))
        r2 = h2.seed_init_defect([("a", _patches(1)), ("b", _patches(2))])
        self.assertEqual(r2["action"], "init_defect_seed_rolled_back")
        self.assertEqual(len(h2.defect_bank.samples), 0)   # 整批撤销


class TestMaskedPatches(unittest.TestCase):
    def test_mask_to_patches(self):
        f = torch.randn(1, D, G, G)
        m = np.zeros((60, 60), np.uint8)
        m[:20, :20] = 255                              # 覆盖左上 2×2 网格
        out = _masked_defect_patches([{"path": "d.png", "feats": f, "tile_masks": [m]}],
                                     min_k=4)
        self.assertEqual(len(out), 1)
        p, pf = out[0]
        self.assertEqual((p, pf.shape, pf.dtype), ("d.png", (4, D), np.float16))
        np.testing.assert_allclose(pf[0], f[0, :, 0, 0].half().numpy())
        none = _masked_defect_patches([{"path": "n.png", "feats": f, "tile_masks": [None]},
                                       {"path": "z.png", "feats": None, "tile_masks": [m]}])
        self.assertEqual(none, [])


if __name__ == "__main__":
    unittest.main()
