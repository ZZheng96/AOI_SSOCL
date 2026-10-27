"""AC-065 模拟定位验收：已知 bbox 的合成缺陷 → 引擎定位链（build_heatmap_mask →
boxes_from_mask → finalize_regions，与 OfflineEngine._hier 同顺序）→ IoU/实例召回/误定位。

- 场景：常规 / 多尺度（不同 H,W 经 work_side 缩放）/ 相邻分离 / 小缺陷 / 边界裁剪 / top-k。
- 热图模拟 patch 级输出：GT 框在 G×G 网格上按面积覆盖率栅格化 + 高斯模糊 + 噪声（两槽位，
  第二槽位噪声更大），与真实热图“粗网格、边缘模糊”的特性一致。
- 后处理只在 dev 切分网格搜索并 select_postproc 冻结；test 切分只评估、不得调参
  （对 test 调 select_postproc 必须报错）。冻结版本写入 review/task/final_sim/raw/。
"""
import json
import os
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

from algo.decision import localization as L
from algo.decision.hierarchical import boxes_from_mask, build_heatmap_mask

OUT = Path(r"d:\CGAIC\AOI_SSOCL\review\task\final_sim\raw")
GRID = 32                      # 模拟 patch 级热图网格
SCEN = ("normal", "multiscale", "adjacent", "small", "boundary", "topk")


def _gt_boxes(rng, scen, H, W):
    cell = max(H, W) / GRID
    out = []

    def box(cx, cy, bw, bh):
        x0, y0 = max(0, int(cx - bw / 2)), max(0, int(cy - bh / 2))
        return [x0, y0, min(W, int(x0 + bw)), min(H, int(y0 + bh))]

    if scen in ("normal", "multiscale"):
        for _ in range(int(rng.integers(1, 4))):
            s = rng.uniform(3, 6) * cell
            out.append(box(rng.uniform(0.15, 0.85) * W, rng.uniform(0.15, 0.85) * H, s, s * rng.uniform(0.7, 1.3)))
    elif scen == "adjacent":                     # 两缺陷间隔约 2.5 格
        s = rng.uniform(3, 4.5) * cell
        cx, cy = rng.uniform(0.3, 0.5) * W, rng.uniform(0.3, 0.7) * H
        out += [box(cx, cy, s, s), box(cx + s + 2.5 * cell, cy, s, s)]
    elif scen == "small":                        # 约 1.5~2.2 格的小缺陷
        for _ in range(2):
            s = rng.uniform(1.5, 2.2) * cell
            out.append(box(rng.uniform(0.2, 0.8) * W, rng.uniform(0.2, 0.8) * H, s, s))
    elif scen == "boundary":                     # 贴边/越界（裁剪到图内）
        s = rng.uniform(3, 5) * cell
        out += [box(s * 0.3, rng.uniform(0.3, 0.7) * H, s, s),
                box(W - s * 0.3, H - s * 0.3, s, s)]
    else:                                        # topk：7 个缺陷，强度递减
        for i in range(7):
            s = 3 * cell
            out.append(box((0.1 + 0.13 * i) * W, (0.25 if i % 2 else 0.7) * H, s, s))
    return [b for b in out if b[2] - b[0] > 2 and b[3] - b[1] > 2]


def _heatmap(rng, gts, H, W, noise, amps=None):
    g = np.zeros((GRID, GRID), np.float32)
    sy, sx = GRID / H, GRID / W
    for k, (x0, y0, x1, y1) in enumerate(gts):
        a = 1.0 if amps is None else amps[k]
        for gy in range(int(y0 * sy), min(GRID, int(np.ceil(y1 * sy)))):
            for gx in range(int(x0 * sx), min(GRID, int(np.ceil(x1 * sx)))):
                ox = min(x1, (gx + 1) / sx) - max(x0, gx / sx)
                oy = min(y1, (gy + 1) / sy) - max(y0, gy / sy)
                cov = max(0.0, ox) * max(0.0, oy) * sx * sy
                g[gy, gx] = max(g[gy, gx], a * cov)
    g = cv2.GaussianBlur(g, (3, 3), 0.6)
    return np.clip(g + rng.normal(0, noise, g.shape).astype(np.float32), 0, None)


def make_split(seed, n_per=6):
    rng = np.random.default_rng(seed)
    items = []
    for scen in SCEN:
        for _ in range(n_per):
            if scen == "multiscale":
                H, W = int(rng.choice([480, 1200, 2500, 3000])), int(rng.choice([640, 1600, 2500, 4096]))
            else:
                H, W = int(rng.choice([600, 1024])), int(rng.choice([800, 1024]))
            gts = _gt_boxes(rng, scen, H, W)
            amps = list(np.linspace(1.0, 0.55, len(gts))) if scen == "topk" else None
            hms = {"a": [_heatmap(rng, gts, H, W, 0.03, amps)],
                   "b": [_heatmap(rng, gts, H, W, 0.08, amps)]}
            items.append({"scen": scen, "H": H, "W": W, "gt": gts, "hms": hms})
    return items


def run_chain(item, cfg):
    """与 OfflineEngine._hier 同顺序的定位链。"""
    params = L.postproc_params(cfg)
    H, W = item["H"], item["W"]
    scale = min(params["work_side"] / max(H, W), 1.0)
    wh, ww = max(1, int(H * scale)), max(1, int(W * scale))
    fused, mask, _ = build_heatmap_mask(item["hms"], (wh, ww), params["mask_thresh"])
    boxes = boxes_from_mask(mask, params["min_area"]) if mask is not None else []
    return L.finalize_regions(boxes, fused, scale, (H, W), params), params


def evaluate(items, cfg, scen=None):
    samples, params = [], None
    for it in items:
        if scen and it["scen"] != scen:
            continue
        regs, params = run_chain(it, cfg)
        samples.append({"pred": [r["bbox"] for r in regs], "gt": it["gt"]})
    return L.eval_localization(samples, iou_thr=0.5, postproc=params["version_id"])


def grid_cfgs():
    for t in (0.3, 0.4, 0.5, 0.6):
        for a in (4, 16, 48):
            yield {"hierarchical": {"mask_thresh": t, "min_area": a, "work_side": 512}}


class FinalSimLocalizationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("[AC-065] 生成合成 dev/test 切分 ...", flush=True)
        cls.dev, cls.test = make_split(1001), make_split(2002)

    def test_1_dev_select_and_freeze(self):
        cands = []
        cfgs = list(grid_cfgs())
        for i, cfg in enumerate(cfgs, 1):
            r = evaluate(self.dev, cfg)
            print(f"  [dev {i}/{len(cfgs)}] {cfg['hierarchical']} recall={r['instance_recall']:.3f} "
                  f"prec={r['precision']:.3f} misloc={r['mislocalization_rate']:.3f}", flush=True)
            cands.append((cfg, r))
        sel = L.select_postproc(cands, "dev", "instance_recall")
        with self.assertRaises(ValueError):
            L.select_postproc(cands, "test")
        with self.assertRaises(ValueError):
            L.select_postproc(cands, "dev", metric="auc")
        frozen = sel["params"]
        fp = L.postproc_params(frozen)
        print(f"[AC-065] dev 冻结: {frozen['hierarchical']} version={fp['version_id']}", flush=True)
        # test 只评估冻结版本
        tr = evaluate(self.test, frozen)
        per = {s: evaluate(self.test, frozen, s) for s in SCEN}
        default = evaluate(self.test, {})
        for s, r in per.items():
            print(f"  [test {s}] recall={r['instance_recall']:.3f} prec={r['precision']} "
                  f"misloc={r['mislocalization_rate']} miou={r['mean_iou']}", flush=True)
        self.assertGreaterEqual(tr["instance_recall"], 0.8)
        self.assertLessEqual(tr["mislocalization_rate"], 0.1)
        self.assertGreaterEqual(tr["mean_iou"], 0.5)
        self.assertGreaterEqual(per["adjacent"]["instance_recall"], 0.8)    # 相邻未粘连
        self.assertGreaterEqual(per["boundary"]["instance_recall"], 0.8)
        self.assertGreaterEqual(per["multiscale"]["instance_recall"], 0.8)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "ac065_loc_frozen.json").write_text(json.dumps({
            "selected_on": "dev(seed=1001)", "frozen_cfg": frozen, "frozen_postproc": fp,
            "dev_result": sel["result"], "test_result(seed=2002)": tr,
            "test_per_scenario": per, "test_default_cfg": default,
            "dev_grid": [{"cfg": c["hierarchical"], **r} for c, r in cands]},
            ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    def test_2_boundary_clip_and_coords(self):
        print("[AC-065] 边界裁剪/坐标空间", flush=True)
        for it in self.test:
            regs, _ = run_chain(it, {})
            for r in regs:
                x0, y0, x1, y1 = r["bbox"]
                self.assertTrue(0 <= x0 <= x1 <= it["W"] and 0 <= y0 <= y1 <= it["H"], (it["scen"], r["bbox"]))

    def test_3_topk_truncation_by_peak(self):
        print("[AC-065] top-k 按峰值截断", flush=True)
        items = [it for it in self.test if it["scen"] == "topk"]
        for it in items:
            full, _ = run_chain(it, {"hierarchical": {"mask_thresh": 0.3}})
            k3, p = run_chain(it, {"hierarchical": {"mask_thresh": 0.3, "top_k": 3}})
            self.assertGreater(len(full), 3)
            self.assertEqual(len(k3), 3)
            self.assertEqual([r["region_id"] for r in k3], ["r0", "r1", "r2"])
            self.assertEqual([r["bbox"] for r in k3], [r["bbox"] for r in full[:3]])
            pk = [r["score_peak"] for r in k3]
            self.assertEqual(pk, sorted(pk, reverse=True))
            self.assertEqual(p["top_k"], 3)

    def test_4_latency_estimate(self):
        print("[AC-065] 后处理耗时估算（2500x2500）", flush=True)
        it = next(i for i in self.test if i["scen"] == "multiscale")
        it = {**it, "H": 2500, "W": 2500}
        t0 = time.perf_counter()
        for _ in range(20):
            run_chain(it, {})
        ms = (time.perf_counter() - t0) / 20 * 1000
        print(f"  定位后处理 {ms:.2f} ms/图", flush=True)
        self.assertLess(ms, 50)


if __name__ == "__main__":
    unittest.main(verbosity=2)
