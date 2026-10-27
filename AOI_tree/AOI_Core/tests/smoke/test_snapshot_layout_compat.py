"""FIX-SNAP（V§63）：旧快照 layout 拟合统计缺失时的兼容加载 smoke。

验证：①缺统计旧快照可加载；②降级状态显式标注；③完整快照行为不变；
④判定路径不使用伪造/默认统计（layout 不打分、不进融合、槽位上无统计属性）。
候选路径与直接 _restore_slot_state 仍严格拒绝（防静默用错统计）。
"""
import json
import os
import pickle
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np

from algo.persist import LAYOUT_REQUIRED_STATS

_CFG = {"tiling": {"mode": "single", "tile_size": 512, "stride": 448},
        "backbone": {"enabled": False}, "slots": {}}


def _imgs(n, seed, defect=False):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        im = rng.normal(40, 5, (96, 96, 3)).clip(0, 255).astype(np.uint8)
        for y, x in ((10, 10), (10, 60), (60, 10), (60, 60)):   # 4 个亮"元件"
            im[y:y + 20, x:x + 20] = 220
        if defect:
            im[60:80, 60:80] = 40                               # 缺件
        out.append(im)
    return out


def _item(img, i):
    return {"path": f"img{i}.png", "img": img, "tiles": [(0, 0) + img.shape[:2]],
            "tile_imgs": [img], "feats": None}


class SnapshotLayoutCompatTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = _imgs(6, 1)
        cls.test = _imgs(2, 2) + _imgs(2, 3, defect=True)

    def _fit_pipe(self):
        from algo.eval.offline import Pipeline
        from algo.fusion.calibrate import CDFCalibrator
        from algo.slots.blob import BlobSlot
        from algo.slots.layout import LayoutSlot
        blob, lay = BlobSlot({}), LayoutSlot({})
        lay.fit({"train_tile_imgs": [[im] for im in self.train]})
        pipe = Pipeline(dict(_CFG), None, [blob, lay])
        self.addCleanup(pipe._cpu_exec.shutdown)
        self.addCleanup(blob._exec.shutdown)
        pipe.calibrators = {}
        for n, s in pipe.slots.items():
            raw = [pipe._slot_image_score(s, _item(im, i))[0] for i, im in enumerate(self.train)]
            pipe.calibrators[n] = CDFCalibrator(n_bins=16).fit(np.asarray(raw))
        pipe.weights = {"blob": 0.6, "layout": 0.4}
        pipe.slot_sanity = {}
        pipe.consensus_gate = pipe.router_gate = None
        pipe.router = pipe.router_slot_names = pipe.open_det = None   # fit 入口才设置
        pipe._score_slots = list(pipe.slots)
        return pipe

    def _save(self, pipe, tmp):
        from algo.persist import save_snapshot
        save_snapshot(pipe, tmp, version=1)

    def _make_legacy(self, tmp, drop=False, rehash=False):
        """模拟 2026-10-11 前快照：layout 只有旧 PCA 键（全 None），无 7 项统计。
        rehash=True：重写完整性哈希（模拟带 manifest 的候选，验证候选路径仍严格）。"""
        p = os.path.join(tmp, "slot_state.pkl")
        with open(p, "rb") as f:
            st = pickle.load(f)
        lay = {k: None for k in ("norm", "centroids", "patterns", "n_components", "pca")}
        if not drop:
            lay.update({k: None for k in LAYOUT_REQUIRED_STATS})
        st["layout"] = lay
        with open(p, "wb") as f:
            pickle.dump(st, f, protocol=pickle.HIGHEST_PROTOCOL)
        mp = os.path.join(tmp, "meta.json")
        with open(mp, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if rehash:
            from algo.persist import _artifact_hashes
            meta["artifact_hashes"] = _artifact_hashes(tmp)
        else:
            meta.pop("artifact_hashes", None)      # 旧快照无完整性哈希
        with open(mp, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)

    def _load(self, tmp, candidate=False):
        from algo.persist import load_snapshot
        from algo.slots.blob import BlobSlot
        from algo.slots.layout import LayoutSlot
        blob = BlobSlot({})
        self.addCleanup(blob._exec.shutdown)
        with patch("algo.factory.build", return_value=(None, [blob, LayoutSlot({})])), \
                patch.dict("os.environ", {"AOI_PICKLE_EXTRA_ROOTS": str(tmp)}), \
                warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            p = load_snapshot(str(tmp), "cpu", candidate=candidate) if candidate \
                else load_snapshot(str(tmp), "cpu")
        self.addCleanup(p._cpu_exec.shutdown)
        return p, [str(w.message) for w in ws]

    def test_full_snapshot_unchanged(self):
        pipe = self._fit_pipe()
        ref = [pipe._record(_item(im, i)) for i, im in enumerate(self.test)]
        with tempfile.TemporaryDirectory() as tmp:
            self._save(pipe, tmp)
            p, _ = self._load(tmp)
        self.assertEqual(p.degraded_slots, {})
        self.assertEqual(p.weights, {"blob": 0.6, "layout": 0.4})
        self.assertIn("layout", p._score_slots)
        for k in LAYOUT_REQUIRED_STATS:
            self.assertEqual(getattr(p.slots["layout"], k), getattr(pipe.slots["layout"], k))
        for i, im in enumerate(self.test):
            r = p._record(_item(im, i))
            self.assertAlmostEqual(r["fused"], ref[i]["fused"], places=9)
            self.assertEqual(r["slot_scores"], ref[i]["slot_scores"])
            self.assertEqual(r["decision_trace"][0]["degraded_slots"], {})

    def test_legacy_snapshot_degrades_explicitly(self):
        for drop in (False, True):
            with self.subTest(drop=drop):
                pipe = self._fit_pipe()
                with tempfile.TemporaryDirectory() as tmp:
                    self._save(pipe, tmp)
                    self._make_legacy(tmp, drop=drop)
                    p, msgs = self._load(tmp)
                info = p.degraded_slots["layout"]
                self.assertTrue(info["layout_degraded"])
                self.assertEqual(info["layout_stats_source"], "missing_in_snapshot")
                self.assertEqual(info["missing"], list(LAYOUT_REQUIRED_STATS))
                self.assertIn("重新 prepare", info["action"])
                self.assertEqual(info["weights_before"], {"blob": 0.6, "layout": 0.4})
                self.assertTrue(any("快照降级加载" in m for m in msgs))
                # 判定路径不使用伪造统计
                self.assertNotIn("layout", p._score_slots)
                self.assertEqual(p.weights["layout"], 0.0)
                self.assertAlmostEqual(sum(p.weights.values()), 1.0)
                for k in LAYOUT_REQUIRED_STATS:
                    self.assertFalse(hasattr(p.slots["layout"], k), k)
                self.assertNotIn("layout", p._prefetch_cpu(_item(self.test[0], 0)))
                for i, im in enumerate(self.test):
                    r = p._record(_item(im, i))
                    self.assertEqual(set(r["slot_scores"]), {"blob"})
                    self.assertAlmostEqual(r["fused"], r["slot_scores"]["blob"], places=9)
                    tr = r["decision_trace"][0]
                    self.assertEqual(tr["degraded_slots"]["layout"]["layout_stats_source"],
                                     "missing_in_snapshot")
                    self.assertNotIn("layout", tr["effective_weights"])

    def test_strict_paths_still_reject(self):
        from algo.persist import LayoutStatsMissing, _restore_slot_state
        from algo.slots.layout import LayoutSlot
        with self.assertRaises(ValueError):
            _restore_slot_state(LayoutSlot({}), {"norm": None})
        pipe = self._fit_pipe()
        with tempfile.TemporaryDirectory() as tmp:
            self._save(pipe, tmp)
            self._make_legacy(tmp, rehash=True)
            with self.assertRaises(LayoutStatsMissing):
                self._load(tmp, candidate=True)

    def test_all_slots_degraded_refuses(self):
        from algo.eval.offline import Pipeline
        from algo.fusion.calibrate import CDFCalibrator
        from algo.slots.layout import LayoutSlot
        lay = LayoutSlot({})
        lay.fit({"train_tile_imgs": [[im] for im in self.train]})
        pipe = Pipeline(dict(_CFG), None, [lay])
        self.addCleanup(pipe._cpu_exec.shutdown)
        pipe.calibrators = {"layout": CDFCalibrator(n_bins=16).fit(np.arange(6.0))}
        pipe.weights, pipe.slot_sanity = {"layout": 1.0}, {}
        pipe.consensus_gate = pipe.router_gate = None
        from algo.persist import load_snapshot
        with tempfile.TemporaryDirectory() as tmp:
            self._save(pipe, tmp)
            self._make_legacy(tmp)
            with patch("algo.factory.build", return_value=(None, [LayoutSlot({})])), \
                    patch.dict("os.environ", {"AOI_PICKLE_EXTRA_ROOTS": str(tmp)}), \
                    warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with self.assertRaises(ValueError):
                    load_snapshot(str(tmp), "cpu")


if __name__ == "__main__":
    unittest.main()
