"""B 块（AC-010~017）特征注册表 / 组启用 / grouped 评分 / 解释 / 签名 smoke。"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


def _imgs(n, seed, defect=False):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        im = (rng.normal(120, 6, (64, 64, 3))).clip(0, 255).astype(np.uint8)
        if defect:
            y, x = rng.integers(8, 40, 2)
            im[y:y + 16, x:x + 16] = 250
        out.append(im)
    return out


class RegistryTest(unittest.TestCase):
    def test_slices_categories_and_trace(self):
        from algo.features import registry as freg
        from algo.vendor.traditional import TraditionalFeatureExtractor
        self.assertEqual(len(freg.TRAD_GROUPS), 15)
        self.assertEqual(sum(g["dim"] for g in freg.TRAD_GROUPS), 200)
        self.assertEqual(freg.TRAD_DIM, 200)
        self.assertEqual(set(freg.by_category()), set(freg.CATEGORIES))
        self.assertTrue(all(freg.by_category()[c] for c in freg.CATEGORIES))
        for g in freg.TRAD_GROUPS:
            for k in ("key", "version", "fn", "params", "slice", "views", "enabled", "profile"):
                self.assertIn(k, g)
        ext = TraditionalFeatureExtractor()
        v = ext.extract(_imgs(1, 0)[0])
        self.assertEqual(v.shape, (200,))
        self.assertEqual(freg.check_trace(ext.last_trace), [])
        self.assertEqual(freg.BACKBONE["key"], "dino_patch")
        with self.assertRaises(ValueError):
            freg.resolve_groups(["nope"])
        with self.assertRaises(ValueError):
            freg.resolve_groups([])
        self.assertEqual(freg.dim_index(["surf"], 172), [])
        self.assertEqual(len(freg.registry_hash()), 64)


class TradGroupTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = _imgs(8, 1)
        cls.good = _imgs(3, 2)
        cls.bad = _imgs(4, 3, defect=True)

    def _fit(self, cfg, defects=None):
        from algo.slots.trad import TradSlot
        s = TradSlot(cfg)
        s.fit({"train_imgs": self.train, "train_paths": [f"n{i}.png" for i in range(8)],
               "defect_imgs": defects or []})
        return s

    def test_legacy_numeric_unchanged(self):
        s = self._fit({"topk": 3, "zclip": 8.0})
        im = self.bad[0]
        v = s._z(s._vec(im), s.norm_w)
        ref = float(np.sort(np.linalg.norm(s.bank_w - v, axis=1))[1:4].mean())
        self.assertAlmostEqual(s.score_image(im)[0], ref, places=6)
        self.assertEqual(s.groups, list(__import__("algo.features.registry", fromlist=["x"]).GROUP_KEYS))

    def test_subset_and_grouped(self):
        sub = self._fit({"groups": ["color", "gray"]})
        self.assertEqual(sub.groups, ["gray", "color"])
        self.assertIsNotNone(sub.score_image(self.bad[0]))
        g = self._fit({"scoring": "grouped"})
        self.assertTrue(all(w in (0.0, 1.0) for w in g.group_weight.values()))
        self.assertTrue(all(sc > 0 for sc in g.group_scale.values()))
        sg = [g.score_image(im)[0] for im in self.good]
        sb = [g.score_image(im)[0] for im in self.bad]
        self.assertGreater(np.mean(sb), np.mean(sg))
        with self.assertRaises(ValueError):
            self._fit({"scoring": "bogus"})

    def test_auto_dev_only(self):
        few = self._fit({"groups": "auto"}, self.bad[:1])
        self.assertEqual(few.selection_report["auto"]["status"], "insufficient_dev")
        a = self._fit({"groups": "auto", "scoring": "grouped"}, self.bad)
        rep = a.selection_report
        self.assertIn(rep["auto"]["status"], ("selected", "no_gain"))
        self.assertIn("single_auroc", rep["auto"])
        self.assertIn("禁止 test", rep["data"])
        self.assertTrue(a.groups)

    def test_explain(self):
        s = self._fit({"scoring": "grouped"})
        e = s.explain(self.bad[0])
        self.assertEqual(e["status"], "available")
        self.assertIn("非缺陷概率", e["score_semantics"])
        self.assertTrue(e["group_breakdown"] and e["top_dims"] and e["neighbors"])
        self.assertTrue(e["neighbors"][0]["path"].startswith("n"))
        for d in e["top_dims"]:
            self.assertIn("normal_p95", d)

    def test_persist_roundtrip_and_stale(self):
        from algo.eval.offline import Pipeline
        from algo.persist import save_snapshot, load_snapshot
        from algo.slots.trad import TradSlot
        s = self._fit({"groups": ["color", "gray", "edge"], "scoring": "grouped"})
        pipe = Pipeline({"tiling": {}, "backbone": {"enabled": False}}, None, [s])
        self.addCleanup(pipe._cpu_exec.shutdown)
        pipe.weights, pipe.slot_sanity = {"trad": 1.0}, {}
        pipe.consensus_gate = pipe.router_gate = None
        with tempfile.TemporaryDirectory() as tmp:
            save_snapshot(pipe, tmp, version=1)

            def load(cfg):
                with patch("algo.factory.build", return_value=(None, [TradSlot(cfg)])), \
                        patch.dict("os.environ", {"AOI_PICKLE_EXTRA_ROOTS": str(tmp)}):
                    p = load_snapshot(str(tmp), "cpu")
                self.addCleanup(p._cpu_exec.shutdown)
                return p
            r = load({})
            rs = r.slots["trad"]
            self.assertEqual(rs.groups, s.groups)
            self.assertEqual(rs.scoring, "grouped")
            self.assertAlmostEqual(rs.score_image(self.bad[0])[0], s.score_image(self.bad[0])[0], places=6)
            self.assertEqual(r.feature_representation["status"], "current")
            with patch("algo.features.registry.registry_hash", return_value="0" * 64):
                r2 = load({})
            self.assertEqual(r2.feature_representation["status"], "stale")
            self.assertIn("registry_sha256", r2.feature_representation["drift"])
            # AC-035：完整快照 manifest——code hash/组件清单/缓存盘点/RNG
            import json, os
            from algo.persist import code_sha256, representation_drift, SNAPSHOT_COMPONENTS
            with open(os.path.join(tmp, "meta.json"), encoding="utf-8") as f:
                meta = json.load(f)
            self.assertEqual(meta["code_sha256"], code_sha256())
            self.assertEqual(set(meta["components"]), set(SNAPSHOT_COMPONENTS))
            self.assertTrue(meta["components"]["sem_and_slots"]["present"])
            self.assertTrue(meta["components"]["calibration"]["present"])
            self.assertFalse(meta["components"]["normal_bank"]["present"])   # 未启 SSOCL
            self.assertIn("backbone_feature_cache", meta["cache_inventory"])
            self.assertIn("policy", meta["rng"])
            self.assertFalse(r.code_drift)
            self.assertEqual(r.backbone_representation["status"], "current")
            self.assertEqual(representation_drift({"backbone_sha256": "a"}, "b"), ["backbone_sha256"])
            self.assertEqual(representation_drift({"backbone_sha256": "a"}, "a"), [])
            self.assertEqual(representation_drift({}, "b"), [])   # 旧快照不误报

    def test_group_profile(self):
        """AC-012：组级档案字段齐全；legacy 全组分数与生产 score_image 一致。"""
        import json
        from algo.eval.contribution import trad_group_profile
        imgs = self.good + self.bad
        labels = [0] * 3 + [1] * 4
        types = [None] * 3 + ["blob"] * 4
        for cfg in ({}, {"scoring": "grouped"}):
            s = self._fit(cfg)
            p = trad_group_profile(s, imgs, labels, types, category="syn", n_boot=50)
            self.assertEqual(p["status"], "available")
            json.dumps(p)
            prod = [s.score_image(im)[0] for im in imgs]
            from algo.eval.contribution import _auc
            self.assertAlmostEqual(p["full_auroc"], _auc(labels, prod), places=6)
            for k in ("registry_sha256", "signature", "category", "n_normal", "n_defect",
                      "views", "created_at", "data_source", "config"):
                self.assertIn(k, p["meta"])
            g = next(v for v in p["groups"].values() if v["enabled"])
            for k in ("single_auroc", "single_ci95", "direction", "fixed_loo_drop",
                      "rebuild_loo_drop", "permutation_drop", "redundancy", "extract_ms",
                      "by_type", "version", "category"):
                self.assertIn(k, g)
            self.assertIn("blob", g["by_type"])
            self.assertTrue(p["cluster_loo"])
        bad = trad_group_profile(s, imgs[:3], [0, 0, 0], [None] * 3)
        self.assertEqual(bad["status"], "insufficient")


if __name__ == "__main__":
    unittest.main()
