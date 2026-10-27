"""A 块（AC-001~009）smoke：视图模式、块尺寸、误报预算、并集/副作用、预算调度、
LocalBranch 拟合/打分/持久化/签名校验、Pipeline 叠加层与定位来源。假 backbone，不依赖 GPU/DINO。"""
import json
import os
import tempfile
import types
import unittest

import numpy as np
import torch

from algo.common.resolution import ResolutionProfile, select_view_mode
from algo.common.tiling import tile_size_bounds, compute_local_tiles, tile_labels_by_area
from algo.decision.branch_budget import (sidak_block_alpha, resolve_budget, fit_union_thresholds,
                                         union_decide, side_effects, BudgetScheduler,
                                         tile_priority_from_heatmap)
from algo.decision.local_branch import LocalBranch, resolve_tile_size, view_signature, mask_boxes


class FakeBackbone:
    """每 tile → (8,G,G) 特征：按 patch 区域像素均值/方差构造，亮块=异常。"""
    device_ = "cpu"
    G = 4

    def extract_tiles(self, tile_imgs, batch=16):
        out = []
        for t in tile_imgs:
            g = t.mean(axis=2).astype(np.float32) / 255.0
            h, w = g.shape
            f = np.zeros((8, self.G, self.G), np.float32)
            for i in range(self.G):
                for j in range(self.G):
                    p = g[i * h // self.G:(i + 1) * h // self.G, j * w // self.G:(j + 1) * w // self.G]
                    m = float(p.mean())
                    f[:, i, j] = [1.0, m, m * m, float(p.std()), 0.5, 1 - m, 0.2, m * 0.5]
            out.append(torch.from_numpy(f))
        return torch.stack(out, 0)


def _cfg(**local):
    lc = {"enabled": True, "tile_size": 64, "stride": 64, "cal_holdout": 0.4,
          "slot": {"coreset_size": 8, "tile_topk_ratio": 0.25}}
    lc.update(local)
    return {"seed": 0, "backbone": {"name": "fake", "grid": 4},
            "tiling": {"local": lc}, "calibration": {"cdf_bins": 16},
            "decision": {"tau_quantile": 0.9}}


def _normal(seed):
    rng = np.random.RandomState(seed)
    return np.clip(100 + rng.randn(128, 128, 3) * 3, 0, 255).astype(np.uint8)


def _defect(seed):
    img = _normal(seed)
    img[70:90, 70:90] = 250
    m = np.zeros((128, 128), np.uint8)
    m[70:90, 70:90] = 255
    return img, m


class ViewModeTest(unittest.TestCase):
    def test_modes(self):
        sz = (2500, 2500)
        p = ResolutionProfile(domain_shift=True, min_defect_px=10)
        self.assertEqual(select_view_mode(p, sz)["mode"], "domain_first")
        p = ResolutionProfile(min_defect_px=50)
        self.assertEqual(select_view_mode(p, sz, train_auroc=0.6)["mode"], "domain_first")
        self.assertEqual(select_view_mode(ResolutionProfile(), sz)["mode"], "profile_missing")
        self.assertEqual(select_view_mode(ResolutionProfile(min_defect_px=400), sz)["mode"], "single")
        r = select_view_mode(ResolutionProfile(min_defect_px=40), sz)
        self.assertEqual(r["mode"], "global_local")
        self.assertEqual(r["bounds"]["status"], "ok")
        self.assertLessEqual(r["bounds"]["lo"], r["bounds"]["hi"])
        # 最大缺陷过大 → 下界 > 上界 → global_only
        r = select_view_mode(ResolutionProfile(min_defect_px=20, max_defect_px=900), sz)
        self.assertEqual(r["mode"], "global_only")

    def test_profile_from_cfg(self):
        p = ResolutionProfile.from_cfg({"tiling": {"profile": {"min_defect_px": 12, "foo": 1}}})
        self.assertEqual(p.min_defect_px, 12)
        self.assertEqual(p.extra, {"foo": 1})


class TileSizeTest(unittest.TestCase):
    def test_bounds_direction(self):
        b = tile_size_bounds(40, grid=32, min_patches=2.0)
        self.assertEqual(b["hi"], 640)          # 40*32/2
        self.assertEqual(b["lo"], 40)           # max(1*32, 40)
        self.assertEqual(tile_size_bounds(None)["status"], "profile_missing")

    def test_resolve(self):
        self.assertEqual(resolve_tile_size({"tile_size": "auto"}),
                         (512, 448, "auto_bounds_missing_candidate_512"))
        t, s, src = resolve_tile_size({"tile_size": "auto"}, {"status": "ok", "lo": 100, "hi": 320})
        self.assertEqual((t, s), (320, 280))
        self.assertTrue(src.startswith("auto_from_bounds"))
        self.assertEqual(resolve_tile_size({"tile_size": 256, "stride": 200}), (256, 200, "cfg"))

    def test_tiles_cover_and_labels(self):
        tiles = compute_local_tiles((300, 300), 128, 100)
        self.assertEqual(tiles, compute_local_tiles((300, 300), 128, 100))   # 确定性
        self.assertEqual(max(t[2] for t in tiles), 300)
        roi_tiles = compute_local_tiles((300, 300), 128, 100, roi=(50, 50, 150, 150))
        self.assertEqual(roi_tiles, [(50, 50, 150, 150)])
        labs = tile_labels_by_area([(0, 0, 100, 100), (0, 100, 100, 200), (0, 90, 100, 190)],
                                   [(10, 10, 50, 50)])
        self.assertEqual(labs[0], 1)
        self.assertEqual(labs[1], 0)


class BudgetTest(unittest.TestCase):
    def test_branch_ratio_nonpositive_tau(self):
        from algo.decision.branch_budget import branch_ratio
        self.assertAlmostEqual(branch_ratio(2.0, 1.0), 2.0)
        self.assertAlmostEqual(branch_ratio(-0.5, -0.5), 1.0)       # 恰在阈值
        self.assertLess(branch_ratio(-0.6, -0.5), 1.0)              # 低于阈值不报警
        self.assertGreater(branch_ratio(-0.4, -0.5), 1.0)
        self.assertEqual(branch_ratio(5.0, float("inf")), 0.0)

    def test_sources_unfrozen(self):
        a, sp, src = resolve_budget({"decision": {"tau_quantile": 0.99}})
        self.assertAlmostEqual(a, 0.01)
        self.assertEqual(sp, {"global": 0.5, "local": 0.5})
        self.assertEqual(src["alpha"], "unfrozen_default_from_tau_quantile")
        self.assertEqual(src["split"], "equal_default_unfrozen")
        a, sp, src = resolve_budget({"decision": {"branch_budget": {
            "image_fpr": 0.02, "split": {"global": 3, "local": 1}}}})
        self.assertEqual((a, sp["global"]), (0.02, 0.75))
        self.assertEqual(src["alpha"], "cfg.decision.branch_budget.image_fpr")

    def test_sidak_and_union_thresholds(self):
        self.assertAlmostEqual(sidak_block_alpha(0.05, 1), 0.05)
        self.assertLess(sidak_block_alpha(0.05, 16), 0.05 / 15)
        rng = np.random.RandomState(0)
        gn, blk, img = rng.rand(200), rng.rand(3200), rng.rand(200)
        th = fit_union_thresholds(gn, blk, img, 16, {"decision": {"tau_quantile": 0.9}})
        self.assertAlmostEqual(th["alpha_global"] + th["alpha_local"], th["alpha"])
        self.assertGreaterEqual(th["tau_local"], th["tau_local_sidak"])
        self.assertGreaterEqual(th["tau_local"], th["tau_local_empirical"])
        r = th["report"]
        self.assertLessEqual(r["fpr_union"], th["alpha"] + 0.03)
        self.assertGreaterEqual(r["fpr_union"], max(r["fpr_global"], r["fpr_local"]))

    def test_union_and_side_effects(self):
        self.assertEqual(union_decide(1.2, 0.1, 1.0, 1.0, 0.9)[0], "anomaly")
        d, u, f = union_decide(0.5, 1.5, 1.0, 1.0, 0.9)
        self.assertEqual((d, f["local"], f["global"]), ("anomaly", True, False))
        self.assertEqual(union_decide(0.95, 0.1, 1.0, 1.0, 0.9)[0], "gray")
        self.assertEqual(union_decide(0.1, None, 1.0, 1.0, 0.9)[0], "normal")
        self.assertEqual(side_effects(True, True), "confirmed")
        self.assertEqual(side_effects(False, True), "added_by_local")
        self.assertEqual(side_effects(True, False), "global_only")
        self.assertEqual(side_effects(True, False, veto=True, local_veto=True), "vetoed")
        self.assertEqual(side_effects(True, False, veto=False, local_veto=True), "global_only")
        self.assertEqual(side_effects(False, False), "none")

    def test_scheduler(self):
        tiles = [(0, x, 10, x + 10) for x in range(0, 100, 10)]
        s = BudgetScheduler(max_tiles=3, rotate_n=1, roi=(0, 0, 10, 10))
        pr = [0.0] * 10
        pr[7], pr[4] = 0.9, 0.8
        sel, rs = s.select(tiles, pr)
        self.assertEqual(rs[0], "roi")
        self.assertEqual(rs[7], "priority")
        self.assertEqual(rs[4], "priority")
        self.assertEqual(sum(v == "rotate" for v in rs.values()), 1)
        self.assertEqual(len(sel), 4)
        # 轮换长期覆盖全部剩余块
        seen = set()
        for _ in range(7):
            _, rs = s.select(tiles, pr)
            seen |= {i for i, v in rs.items() if v == "rotate"}
        self.assertEqual(seen, {1, 2, 3, 5, 6, 8, 9})

    def test_priority_from_heatmap(self):
        heat = np.zeros((10, 10), np.float32)
        heat[8, 8] = 1.0
        pr = tile_priority_from_heatmap(heat, [(0, 0, 50, 50), (50, 50, 100, 100)], (100, 100))
        self.assertEqual(pr, [0.0, 1.0])
        self.assertEqual(tile_priority_from_heatmap(None, [(0, 0, 1, 1)], (1, 1)), [0.0])


class LocalBranchTest(unittest.TestCase):
    def setUp(self):
        self.normals = [_normal(i) for i in range(10)]
        d = [_defect(100 + i) for i in range(2)]
        self.d_imgs, self.d_masks = [x[0] for x in d], [x[1] for x in d]

    def _fit(self, cfg=None):
        lb = LocalBranch(cfg or _cfg(), FakeBackbone(), device="cpu")
        return lb.fit(self.normals, self.d_imgs, self.d_masks, log=lambda s: None)

    def test_pseudo_off_by_default(self):
        with self.assertRaises(ValueError):
            LocalBranch(_cfg(pseudo=True), FakeBackbone(), device="cpu")

    def test_fit_score_report(self):
        lb = self._fit()
        rep = lb.report
        self.assertEqual(rep["n_holdout"] + rep["n_bank"], 10)
        self.assertFalse(rep["same_bank_holdout"])
        self.assertEqual(rep["n_tiles_per_image"], 4)
        self.assertEqual(rep["label_thresholds"]["source"], "candidate_unfrozen")
        self.assertGreater(rep["block_auroc_vs_normal"], 0.9)
        self.assertEqual(rep["coverage_union"], 1.0)
        self.assertEqual(len(lb.normal_image_raw), rep["n_holdout"])
        sn = lb.score(_normal(999))
        sd = lb.score(self.d_imgs[0])
        self.assertEqual(sn["n_total"], 4)
        self.assertGreater(sd["score"], sn["score"])
        self.assertEqual(int(np.argmax(sd["block_scores"])), 3)   # 缺陷在右下块

    def test_budget_mode_and_force_full(self):
        lb = self._fit(_cfg(budget={"mode": "budget", "max_tiles": 1, "rotate_n": 1}))
        out = lb.score(self.d_imgs[0], priority_fn=lambda tiles: [0, 0, 0, 1])
        self.assertIn(3, out["selected"])
        self.assertEqual(out["reasons"][3], "priority")
        self.assertEqual(len(out["selected"]), 2)
        self.assertEqual(len(lb.score(self.d_imgs[0], force_full=True)["selected"]), 4)

    def test_state_roundtrip_and_signature(self):
        lb = self._fit()
        arrays = {k: np.asarray(v) for k, v in lb.state_arrays().items()}
        meta = json.loads(json.dumps(lb.state_meta(), default=str))
        lb2 = LocalBranch(_cfg(), FakeBackbone(), device="cpu").load_state(arrays, meta, log=lambda s: None)
        self.assertTrue(lb2.enabled)
        img = self.d_imgs[1]
        self.assertAlmostEqual(lb2.score(img)["score"], lb.score(img)["score"], places=2)
        # 视图变化（tile_size）→ 签名不一致 → 停用
        lb3 = LocalBranch(_cfg(tile_size=32, stride=32), FakeBackbone(), device="cpu")
        lb3.load_state(arrays, meta, log=lambda s: None)
        self.assertFalse(lb3.enabled)
        self.assertNotEqual(view_signature(_cfg())[0], view_signature(_cfg(tile_size=32))[0])

    def test_persist_load(self):
        from algo.persist import _load_local_branch
        lb = self._fit()
        with tempfile.TemporaryDirectory() as d:
            np.savez_compressed(os.path.join(d, "local_branch.npz"), **lb.state_arrays())
            with open(os.path.join(d, "local_branch.json"), "w", encoding="utf-8") as f:
                json.dump({"state": lb.state_meta(), "view_mode": {"mode": "global_local"},
                           "branch_budget_report": {"mode": "global_local"}}, f, default=str)
            pipe = types.SimpleNamespace(cfg=_cfg(), backbone=FakeBackbone())
            _load_local_branch(pipe, d)
            self.assertIsNotNone(pipe.local_branch)
            self.assertTrue(pipe.local_branch.enabled)
            self.assertEqual(pipe.view_mode["mode"], "global_local")
        with tempfile.TemporaryDirectory() as d:     # 旧快照：无文件 → 仅整图
            pipe = types.SimpleNamespace(cfg=_cfg(), backbone=FakeBackbone())
            _load_local_branch(pipe, d)
            self.assertIsNone(pipe.local_branch)

    def test_mask_boxes(self):
        self.assertEqual(mask_boxes(self.d_masks[0]), [(70, 70, 90, 90)])
        self.assertEqual(mask_boxes(None), [])


class PipelineOverlayTest(unittest.TestCase):
    """_apply_local_branch：并集叠加、副作用分报、灰区/冲突全量重算、否决只降为 gray。"""

    def _pipe(self, lb):
        from algo.eval.offline import Pipeline
        p = Pipeline.__new__(Pipeline)
        p.local_branch, p._last_hms, p.weights = lb, {}, {}
        return p

    def _lb(self, **local):
        normals = [_normal(i) for i in range(10)]
        lb = LocalBranch(_cfg(**local), FakeBackbone(), device="cpu").fit(normals, log=lambda s: None)
        s_def = lb.score(_defect(99)[0], force_full=True)["score"]
        lb.thresholds = {"tau_global": 1.0,
                         "tau_local": 0.5 * (max(lb.normal_image_raw) + s_def),
                         "tau_gray_ratio": 0.9, "sources": {"alpha": "test"}}
        return lb

    def test_added_by_local(self):
        from algo.eval.offline import Pipeline
        lb = self._lb()
        p = self._pipe(lb)
        img, _ = _defect(7)
        rec, tm = {"fused": 0.2}, {}
        d, br = Pipeline._apply_local_branch(p, {"img": img}, rec, "normal", tm)
        self.assertEqual(d, "anomaly")
        self.assertEqual(br["side_effect"], "added_by_local")
        self.assertIn("local", tm)
        self.assertIn("_local_out", rec)
        self.assertEqual(br["view_signature"], lb.sig)
        d, br = Pipeline._apply_local_branch(p, {"img": _normal(55)}, {"fused": 0.2}, "normal")
        self.assertEqual((d, br["side_effect"]), ("normal", "none"))
        d, br = Pipeline._apply_local_branch(p, {"img": img}, {"fused": 1.5}, "anomaly")
        self.assertEqual(br["side_effect"], "confirmed")

    def test_veto_only_to_gray(self):
        from algo.eval.offline import Pipeline
        lb = self._lb(veto=True, veto_ratio=10.0)
        d, br = Pipeline._apply_local_branch(self._pipe(lb), {"img": _normal(56)},
                                             {"fused": 1.5}, "anomaly")
        self.assertEqual((d, br["side_effect"]), ("gray", "vetoed"))

    def test_disabled_passthrough(self):
        from algo.eval.offline import Pipeline
        lb = self._lb()
        lb.enabled = False
        self.assertEqual(Pipeline._apply_local_branch(self._pipe(lb), {"img": _normal(1)},
                                                      {"fused": 0.1}, "normal"), ("normal", None))

    def test_budget_conflict_rescore(self):
        from algo.eval.offline import Pipeline
        lb = self._lb(budget={"mode": "budget", "max_tiles": 1, "rotate_n": 0})
        img, _ = _defect(8)
        p = self._pipe(lb)
        # 热度为 0 → 优先块不确定；全局分处于灰区 → 必须全量重算并检出
        d, br = Pipeline._apply_local_branch(p, {"img": img}, {"fused": 0.95}, "gray")
        self.assertEqual(br["n_total"], 4)
        if br["rescored_full"] is not None:
            self.assertTrue(br["full_coverage"])
        self.assertEqual(d, "anomaly")


if __name__ == "__main__":
    unittest.main()
