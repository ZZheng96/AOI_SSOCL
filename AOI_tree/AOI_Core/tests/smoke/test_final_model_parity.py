"""AC-062/AC-010 最终验收：候选特征实现 vs 生产 TradSlot 八项等价（合成图，TEMP 注册表）。"""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from algo.feature_revision import FeatureSetRevision, PARITY_CHECKS, production_hashes
from algo.features.parity import ProductionCandidate, run_parity


def _img(seed, h=64, w=64):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = 120 + 40 * np.sin(xx / (3 + seed % 4)) + 30 * np.cos(yy / 5.0)
    g = np.clip(base + rng.normal(0, 8, (h, w)), 0, 255).astype(np.uint8)
    return np.stack([g, np.roll(g, 1, 0), np.roll(g, 1, 1)], -1)


TRAIN = [_img(s) for s in range(6)]
PROBE = [_img(100, 300, 280), _img(101, 20, 40), _img(102)]   # 缩小 / 放大 / 原尺寸


class UnseededRngCandidate(ProductionCandidate):
    """模拟 AOI_feature feature_lib：未播种且消耗全局 RNG 的自相似采样。"""

    def extract(self, img):
        rng = np.random.default_rng(np.random.randint(1 << 31))
        v = self.slot.ext.extract(img, rng=rng)
        return v.astype(np.float32), list(self.slot.ext.last_trace)


class NoSkipSelfCandidate(ProductionCandidate):
    """模拟 GroupKNN(skip_self=False)：近邻取 [0:3]。"""

    def neighbors(self, img):
        v = self.slot._z(self.slot._vec(img), self.slot.norm_w)
        d = self.slot._dist(self.slot.bank_w, v)
        order = np.argsort(d)
        return float(np.sort(d)[0:3].mean()), [int(i) for i in order[0:3]]

    def report(self, img):
        r = super().report(img)
        s, nb = self.neighbors(img)
        return {**r, "score": s, "neighbors": nb}


class ParityHarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ev = {}
        for i, (name, cand) in enumerate([("production", ProductionCandidate()),
                                          ("unseeded_rng", UnseededRngCandidate()),
                                          ("no_skip_self", NoSkipSelfCandidate())]):
            print(f"[parity {i + 1}/3] {name} ...", flush=True)
            cls.ev[name] = run_parity(cand, TRAIN, PROBE)
            print(f"  -> {cls.ev[name]['status']} failed={cls.ev[name]['failed_checks']}", flush=True)

    def test_production_self_parity_passes_all_eight(self):
        e = self.ev["production"]
        self.assertEqual(e["status"], "passed")
        self.assertEqual(e["verdict"], "equivalent")
        self.assertEqual(set(e["checks"]), set(PARITY_CHECKS))
        self.assertTrue(all(e["checks"].values()))
        self.assertEqual(e["baseline_implementation_hashes"], production_hashes())
        self.assertRegex(e["protocol_hash"], r"^[0-9a-f]{64}$")
        self.assertGreater(e["numeric"]["n_samples"], 0)
        self.assertLessEqual(e["numeric"]["max_abs_diff"], e["numeric"]["atol"])

    def test_unseeded_rng_is_new_candidate(self):
        e = self.ev["unseeded_rng"]
        self.assertEqual(e["verdict"], "new_candidate")
        self.assertIn("rng", e["failed_checks"])
        self.assertTrue(e["diffs"]["rng"]["global_rng_mutated"])

    def test_no_skip_self_fails_knn_scoring_report(self):
        e = self.ev["no_skip_self"]
        self.assertEqual(e["status"], "failed")
        for k in ("knn_neighbors_self_match", "scoring", "report"):
            self.assertIn(k, e["failed_checks"])
        for k in ("extraction", "preprocess", "rng", "bank_compatibility", "dimension_order_slice"):
            self.assertTrue(e["checks"][k], k)

    def test_global_rng_not_mutated_by_harness(self):
        np.random.seed(7)
        marker = np.random.random()
        np.random.seed(7)
        run_parity(ProductionCandidate(), TRAIN[:4], PROBE[2:])
        self.assertEqual(np.random.random(), marker)

    def test_activation_gate_consumes_evidence(self):
        """AC-010：生产自对照证据可激活；候选证据被 fail-closed 拒绝；生产哈希不变。"""
        with tempfile.TemporaryDirectory(prefix="aoi_final_parity_") as d:
            root = Path(d)
            before = production_hashes()
            for name, should_pass in (("production", True), ("no_skip_self", False),
                                      ("unseeded_rng", False)):
                store = FeatureSetRevision(root / f"{name}.json")
                manifest = {"schema_version": 1, "groups": [{"key": "a", "dim": 2, "slice": [0, 2],
                                                             "params": {"n": name}}],
                            "dimension": 2, "preprocess": {"max_side": 256},
                            "implementation_hashes": {"test": "a" * 64}, "seed": 42,
                            "dataset_split": {"test": [{"sha256": "b" * 64, "label": 0}]},
                            "model_snapshot": {"category": "temp", "version": 1,
                                               "artifact_hashes": {"bank": "c" * 64}}}
                rid = store.register(manifest)
                lineage = {"revision_id": rid, **{k: manifest[k] for k in
                                                  ("seed", "dataset_split", "model_snapshot")}}
                ev = {**lineage, "eval_run_id": "t", "status": "completed", "independent": True,
                      "acceptance": "passed", "auroc": 0.9}
                (root / "eval.json").write_text(json.dumps(ev), encoding="utf-8")
                store.transition(rid, "evaluated", actor="t", evidence_path=root / "eval.json")
                store.transition(rid, "approved", actor="t")
                par = {**self.ev[name], **lineage}
                pp = root / f"parity_{name}.json"
                pp.write_text(json.dumps(par), encoding="utf-8")
                with self.subTest(candidate=name):
                    if should_pass:
                        store.transition(rid, "active", actor="t", parity_path=pp)
                        self.assertEqual(store.list()[rid]["state"], "active")
                    else:
                        with self.assertRaises(ValueError):
                            store.transition(rid, "active", actor="t", parity_path=pp)
                        self.assertEqual(store.list()[rid]["state"], "approved")
            self.assertEqual(before, production_hashes())


if __name__ == "__main__":
    unittest.main()
