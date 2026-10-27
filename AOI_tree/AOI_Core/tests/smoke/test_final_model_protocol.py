"""专家协议/融合最终验收（合成数据，纯函数，不加载模型、不写存储）。
AC-003/004/005（tiling）、AC-006（分支预算）、AC-008/009（三方案）、AC-018/020/021/022/023（CDE）。"""
import unittest

import numpy as np

from algo.common import tiling as T
from algo.decision import branch_budget as BB
from algo.decision.scheme_compare import SCHEMES, compare_schemes
from algo.fusion import cde_report as cr
from algo.fusion.calibrate import CDFCalibrator


def _p(msg):
    print(msg, flush=True)


class TilingProtocolTests(unittest.TestCase):
    def test_ac004_random_sizes_union_and_inverse(self):
        _p("[tiling 1/3] 随机尺寸 / 末行末列回移 / 并集覆盖 / 原图反解")
        rng = np.random.default_rng(0)
        for it in range(60):
            h, w = int(rng.integers(40, 1600)), int(rng.integers(40, 1600))
            ts = int(rng.choice([128, 256, 384, 512]))
            st = int(rng.integers(ts // 2, ts + 1))
            tiles = T.compute_local_tiles((h, w), ts, st)
            with self.subTest(it=it, hw=(h, w), ts=ts, st=st):
                self.assertTrue(tiles)
                self.assertEqual(len(tiles), len(set(tiles)))
                for y1, x1, y2, x2 in tiles:
                    self.assertTrue(0 <= y1 < y2 <= h and 0 <= x1 < x2 <= w)
                    self.assertEqual((y2 - y1, x2 - x1), (min(ts, h), min(ts, w)))  # 小图不产生空块
                self.assertEqual(max(t[2] for t in tiles), h)     # 末行贴边
                self.assertEqual(max(t[3] for t in tiles), w)     # 末列贴边
                # 跨块随机缺陷框：并集完全覆盖
                bx1, by1 = int(rng.integers(0, w - 1)), int(rng.integers(0, h - 1))
                box = (bx1, by1, min(w, bx1 + int(rng.integers(1, 200))),
                       min(h, by1 + int(rng.integers(1, 200))))
                self.assertTrue(T.coverage_check(tiles, [box])[0]["union_ok"])
                # 原图反解：tile 内整框（tile 分辨率 448）→ 原图恰为 tile 本身
                t = tiles[int(rng.integers(len(tiles)))]
                inv = T.tile_box_to_image(t, (0, 0, 448, 448), (448, 448))
                np.testing.assert_allclose(inv, (t[1], t[0], t[3], t[2]))
                cy, cx = T.tile_to_image_coords(t, (0, 0), (32, 32))
                self.assertTrue(t[0] <= cy < t[2] and t[1] <= cx < t[3])

    def test_ac003_roi_and_ac004_bounds(self):
        _p("[tiling 2/3] ROI 覆盖 / 块尺寸上下界（冲突交全局）")
        tiles = T.compute_local_tiles((1000, 1200), 256, 200, roi=(100, 150, 700, 900))
        self.assertTrue(all(100 <= t[0] and t[2] <= 700 and 150 <= t[1] and t[3] <= 900 for t in tiles))
        self.assertTrue(T.coverage_check(tiles, [(150, 100, 900, 700)])[0]["union_ok"])
        self.assertEqual(T.tile_size_bounds(None)["status"], "profile_missing")
        ok = T.tile_size_bounds(16, grid=32, min_patches=2)
        self.assertEqual(ok["status"], "ok")
        self.assertLessEqual(ok["lo"], ok["hi"])
        self.assertEqual(T.tile_size_bounds(4, grid=32, min_patches=2, max_defect_px=200)["status"],
                         "global_only")

    def test_ac005_area_labels_and_context_crop(self):
        _p("[tiling 3/3] 面积占比标签（≥0.25 正 / ≤0.05 负 / 中间丢弃）+ 中心上下文裁切")
        # 占比：0.5 / 0.75 / 0.075（中间丢弃）/ 0
        tiles = [(0, 0, 100, 100), (0, 90, 100, 190), (0, 117, 100, 217), (0, 500, 100, 600)]
        box = (80, 10, 120, 50)        # x1,y1,x2,y2 → 宽 40
        self.assertEqual(T.tile_labels_by_area(tiles, [box]), [1, 1, -1, 0])
        c = T.context_crop((1000, 1000), (480, 480, 520, 520), 64, context_px=16)
        self.assertEqual((c[2] - c[0], c[3] - c[1]), (72, 72))
        self.assertTrue(c[0] <= 480 - 16 and c[2] >= 520 + 16)
        e = T.context_crop((100, 100), (0, 0, 10, 10), 64)
        self.assertEqual(e, (0, 0, 64, 64))


class BranchBudgetTests(unittest.TestCase):
    def test_ac006_union_fpr_within_budget(self):
        _p("[budget] 两层校准（Šidák 块级 + 经验图级）→ 合成正常集并集 FPR ≤ α")
        rng = np.random.default_rng(1)
        n_img, n_tiles = 4000, 9
        cfg = {"decision": {"branch_budget": {"image_fpr": 0.05, "split": {"global": .5, "local": .5}}}}
        for rho in (0.0, 0.8):          # 块独立 / 强相关
            g_cal = rng.normal(0, 1, n_img)
            shared = rng.normal(0, 1, (n_img, 1))
            blk_cal = np.sqrt(rho) * shared + np.sqrt(1 - rho) * rng.normal(0, 1, (n_img, n_tiles))
            r = BB.fit_union_thresholds(g_cal, blk_cal.ravel(), blk_cal.max(1), n_tiles, cfg)
            self.assertEqual(r["sources"]["alpha"], "cfg.decision.branch_budget.image_fpr")
            self.assertAlmostEqual(r["alpha_block_sidak"], 1 - (1 - .025) ** (1 / 9))
            self.assertFalse(r["report"]["small_sample"])
            # 独立测试正常集（不同种子）验证预算
            g_t = rng.normal(0, 1, n_img)
            sh_t = rng.normal(0, 1, (n_img, 1))
            b_t = (np.sqrt(rho) * sh_t + np.sqrt(1 - rho) * rng.normal(0, 1, (n_img, n_tiles))).max(1)
            fpr = float(((g_t >= r["tau_global"]) | (b_t >= r["tau_local"])).mean())
            with self.subTest(rho=rho):
                self.assertLessEqual(r["report"]["fpr_union"], 0.05 + 2.0 / n_img)   # 分位插值粒度
                self.assertLessEqual(fpr, 0.05 + 0.012)     # 二项抽样容差（n=4000，≈2σ）
        _, _, src = BB.resolve_budget({"decision": {"tau_quantile": .99}})
        self.assertEqual(src["alpha"], "unfrozen_default_from_tau_quantile")
        small = BB.fit_union_thresholds(g_cal[:10], blk_cal[:10].ravel(), blk_cal[:10].max(1),
                                        n_tiles, cfg)
        self.assertTrue(small["report"]["small_sample"])


class SchemeCompareTests(unittest.TestCase):
    def test_ac008_ac009_three_schemes(self):
        _p("[schemes] 整图 / 全局+全量局部 / 全局+预算细判 同数据对照 + 副作用分报")
        imgs = [
            {"label": 1, "s_global": .2, "blocks": [.1, .9, .1, .1], "budget_idx": [1], "defect_tiles": [1]},
            {"label": 1, "s_global": .2, "blocks": [.1, .1, .1, .8], "budget_idx": [0], "defect_tiles": [3]},
            {"label": 1, "s_global": .9, "blocks": [.1, .1, .1, .1], "budget_idx": [0], "veto": True},
            {"label": 0, "s_global": .1, "blocks": [.7, .1, .1, .1], "budget_idx": [1]},
            {"label": 0, "s_global": .8, "blocks": [.1, .1, .1, .1], "budget_idx": [0], "veto": True},
            {"label": 0, "s_global": .1, "blocks": [.1, .1, .1, .1], "budget_idx": [0]},
        ]
        r = compare_schemes(imgs, .5, .5, peak_vram_mb={"global_only": 100.0})
        self.assertEqual(set(r["schemes"]), set(SCHEMES))
        go, full, bud = (r["schemes"][k] for k in SCHEMES)
        self.assertEqual((go["n_blocks_total"], full["n_blocks_total"], bud["n_blocks_total"]), (0, 24, 6))
        self.assertAlmostEqual(go["recall"], 1 / 3)
        self.assertAlmostEqual(go["fpr"], 1 / 3)
        # 全量：救回 2 漏检、新增 1 误报、否决全局阳性 → 新增漏检 1、误报确认被撤 1
        self.assertEqual((full["rescued"], full["new_false_alarms"], full["veto_new_misses"]), (2, 1, 1))
        self.assertEqual(r["side_effects"]["global_local_full"],
                         {"added_by_local": 3, "confirmed": 0, "global_only": 0, "vetoed": 2, "none": 1})
        self.assertEqual(full["defect_tile_coverage"], 1.0)
        # 预算：仅救回 1，覆盖 0.5，无新增误报
        self.assertEqual((bud["rescued"], bud["new_false_alarms"], bud["defect_tile_coverage"]), (1, 0, .5))
        self.assertEqual(r["deltas_vs_global"]["global_local_full"]["recall"], round(2 / 3 - 1 / 3, 4))
        self.assertEqual(r["deltas_vs_global"]["global_local_budget"]["fpr"], round(0 - 1 / 3, 4))  # 误报确认被撤
        self.assertEqual(go["vram_source"], "measured")
        self.assertEqual(full["vram_source"], "not_measured")
        self.assertIsNone(full["peak_vram_mb"])


class _Cal:
    def fit(self, s):
        self.e = np.sort(np.asarray(s, float))
        return self

    def transform(self, s):
        return np.searchsorted(self.e, np.asarray(s, float), side="right") / len(self.e)


class CdeFinalTests(unittest.TestCase):
    def test_ac018_tail_stability_and_self_match_bias(self):
        _p("[cde 1/4] CDF 契约：尾部 bootstrap std 随 n 收敛 / 小样本 / 自匹配偏差")
        rng = np.random.default_rng(2)
        small, big = rng.normal(0, 1, 15), rng.normal(0, 1, 3000)
        cal_big = CDFCalibrator(256)
        cal_big.fit(big)
        c = cr.calibration_contract({"s": small, "b": big}, {"s": _Cal().fit(small), "b": cal_big},
                                    256, isolation="holdout",
                                    holdout_scores={"b": rng.normal(-0.5, 1, 500)}, n_boot=100)
        s, b = c["slots"]["s"], c["slots"]["b"]
        self.assertTrue(s["small_sample"])
        self.assertFalse(b["small_sample"])
        self.assertLess(b["tail"]["p99_boot_std"], s["tail"]["p99_boot_std"])
        self.assertLess(b["tail"]["p95_boot_std"], s["tail"]["p95_boot_std"])
        self.assertLess(b["self_match_bias"], 0)       # 自匹配侧偏低 → 阈值偏松风险可见
        self.assertIn("非缺陷概率", b["semantics"])
        self.assertFalse(c["self_match_risk"])
        self.assertEqual(b["method"], "cdf")
        self.assertEqual(b["clip"], [-1.0, 2.0])

    def test_ac020_unverified_slot_excluded(self):
        _p("[cde 2/4] 固定融合基线：未验证槽（sanity 缺/≤0.5）不参与，硬规则单列")
        rng = np.random.default_rng(3)
        cn = {"a": rng.uniform(0, .6, 50), "b": rng.uniform(0, 1, 50), "c": rng.uniform(0, 1, 50)}
        cd = {"a": rng.uniform(.4, 1, 50), "b": rng.uniform(0, 1, 50), "c": rng.uniform(0, 1, 50)}
        fb = cr.fusion_baselines(cn, cd, ["a", "b", "c"], {"equal": {"a": 1, "b": 1, "c": 1},
                                                          "sanity": {"a": .9, "b": .5, "c": .2}},
                                 "equal", slot_sanity={"a": .9, "b": .5})
        self.assertEqual(fb["slots_used"], ["a"])
        self.assertEqual(set(fb["unverified_slots"]), {"b", "c"})
        self.assertEqual(set(fb["baselines"]), {"equal", "sanity", "hard_rule_max"})
        for row in ("equal", "sanity"):
            self.assertEqual(fb["baselines"][row]["weights"], {"a": 1.0})
        self.assertGreater(fb["baselines"]["equal"]["auc"], .7)

    def test_ac021_ac023_router_admission_three_splits(self):
        _p("[cde 3/4] 软路由准入：分层三集 / 不胜基线则 trace_only / 熵与过拟合报告")
        rng = np.random.default_rng(4)
        neg = rng.uniform(0, 1, (60, 3))
        pos = rng.uniform(0, 1, (30, 3))
        pos[:, 0] += 0.6                          # 只有 a 有效；固定权重全押 a → 路由难以再胜 margin
        rep = cr.router_admission(neg, pos, ["a", "b", "c"], {"a": 1.0, "b": 0, "c": 0}, {"epochs": 20})
        self.assertEqual(rep["n_params"], 3 * 3 + 3)
        self.assertEqual(rep["split_sizes"], {"train": {"neg": 36, "pos": 18},
                                              "val": {"neg": 12, "pos": 6},
                                              "test": {"neg": 12, "pos": 6}})
        self.assertEqual(rep["decision"], "trace_only")
        self.assertFalse(rep["permitted"])
        for k in ("norm_entropy", "overfit_gap", "collapsed"):
            self.assertIn(k, rep)
        self.assertTrue(0 <= rep["norm_entropy"] <= 1)
        sp = cr.stratified_split(60, 30, seed=1)
        self.assertEqual(sorted(np.concatenate([v[0] for v in sp.values()]).tolist()), list(range(60)))
        self.assertEqual(sorted(np.concatenate([v[1] for v in sp.values()]).tolist()), list(range(30)))

    def test_ac022_domain_shift_metrics(self):
        _p("[cde 4/4] 域差：中位漂移/IQR + FPR 膨胀，动态路由标 not_integrated")
        rng = np.random.default_rng(5)
        ref = rng.normal(0, 1, 500)
        tau = float(np.quantile(ref, .95))
        same = cr.domain_shift(ref, rng.normal(0.05, 1, 300), tau, .05)
        self.assertEqual(same["status"], "normal_drift")
        self.assertLess(same["fpr_inflation"], 2.0)
        fpr_only = cr.domain_shift(ref, rng.normal(0, 2.0, 300), tau, .05)   # 中位不动，尾部膨胀
        self.assertLess(abs(fpr_only["median_shift_iqr"]), .5)
        self.assertEqual(fpr_only["status"], "shift")
        moved = cr.domain_shift(ref, rng.normal(1.5, 1, 300), tau, .05)
        self.assertEqual(moved["status"], "shift")
        self.assertTrue(moved["mechanisms"]["dynamic_routing"].startswith("not_integrated"))


if __name__ == "__main__":
    unittest.main()
