"""定位基线测量（U64 阶段 1，2026-08-16）：异常热力图亮部 vs 标注缺陷框的对齐质量

用户指令：把 GYU-DET 缺陷框当作 GT 检验检测效果——异常热力图的亮部应多出现在
标注的缺陷框内。本脚本对 test 缺陷图逐槽位计算：
  1. 框内能量占比 box_energy：热力图落在标注框内的分数占比（定位好 → 应高）
  2. HitRate@框：热力图全局 top-1% 亮区与任一标注框的重合度（IoU>0 命中率）
  3. 逐槽位汇总 + fused（等权/实际权重）

信息合法性（U64 界定）：标注框属"有缺陷框标注"信息层次，仅用于**检验/评价**
（test 只验不选），不参与 fit/权重计算——诚实边界与 U61 一致。
"""
import os
import sys
import glob
import json
import time
import yaml
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from m0_baseline import build
from src.eval.offline import Pipeline
from src.data import gyudet
from src.common.io import load_image

CFG = os.path.join(os.path.dirname(__file__), "..", "configs", "m0_gyudet_sheadA.yaml")
IMG_ROOT = r"D:\CGAIC\data_origin\GYU-DET\test\images"
LBL_ROOT = r"D:\CGAIC\data_origin\GYU-DET\test\labels"


def load_boxes(lbl_path, grid):
    """YOLO 归一化 (cls cx cy w h) -> 网格坐标框列表 [(gx0, gy0, gx1, gy1)]"""
    boxes = []
    if not os.path.exists(lbl_path):
        return boxes
    for line in open(lbl_path, encoding="utf-8"):
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cx, cy, w, h = [float(x) for x in parts[1:5]]
        x0, y0 = (cx - w / 2) * grid, (cy - h / 2) * grid
        x1, y1 = (cx + w / 2) * grid, (cy + h / 2) * grid
        boxes.append((x0, y0, x1, y1))
    return boxes


def hit_rate_and_energy(hm, boxes, top_ratio=0.01, energy_thresh=0.3):
    """hm: (G,G) float32；boxes: 网格框列表。
    返回 (hit_rate, box_energy, has_box)"""
    if not boxes or hm is None:
        return None, None, False
    G = hm.shape[0]
    tot = float(hm.sum())
    if tot <= 0:
        return 0.0, 0.0, True
    # HitRate：全局 top top_ratio 亮区与任一框重合
    k = max(1, int(G * G * top_ratio))
    idx = np.argpartition(hm.ravel(), -k)[-k:]
    ys, xs = np.unravel_index(idx, (G, G))
    hit = 0
    for x, y in zip(xs, ys):
        for (x0, y0, x1, y1) in boxes:
            if x0 <= x < x1 and y0 <= y < y1:
                hit += 1
                break
    hit_rate = hit / k
    # 框内能量占比
    energy = 0.0
    for (x0, y0, x1, y1) in boxes:
        x0i, y0i = max(0, int(x0)), max(0, int(y0))
        x1i, y1i = min(G, int(np.ceil(x1))), min(G, int(np.ceil(y1)))
        if x1i > x0i and y1i > y0i:
            energy += float(hm[y0i:y1i, x0i:x1i].sum())
    return hit_rate, energy / tot, True


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    with open(CFG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = "cuda"
    bundle = gyudet.load_gyudet(cfg["datasets"]["gyu_det"],
                                cfg["protocol"]["n_init_normal"],
                                cfg["protocol"]["n_init_defect"], cfg["seed"],
                                max_test=150)
    backbone, slots = build(cfg, device)
    pipe = Pipeline(cfg, backbone, slots)
    pipe.fit(bundle)
    print(f"[loc] fit 完成，槽位: {list(pipe.slots)}", flush=True)

    imgs = sorted(glob.glob(os.path.join(IMG_ROOT, "*.jpg")) +
                  glob.glob(os.path.join(IMG_ROOT, "*.png")))
    # U64：用户指令——随机取少数（默认 30）张加快迭代，固定 seed 可复现
    rng = np.random.default_rng(42)
    if len(imgs) > 200:
        imgs = rng.choice(imgs, 200, replace=False).tolist()
    agg = {}   # slot -> {hit:[], energy:[]}
    fused_hit, fused_energy, fused_has = [], [], []
    n_done, n_box = 0, 0
    for p in imgs:
        lbl = os.path.join(LBL_ROOT, os.path.splitext(os.path.basename(p))[0] + ".txt")
        boxes = load_boxes(lbl, grid=cfg["backbone"]["grid"])
        if not boxes:
            continue
        n_box += 1
        r = pipe.predict(p)
        hms = getattr(pipe, "_last_hms", {}) or {}
        if n_done == 0:
            for name, hm_list in hms.items():
                hm = hm_list[0] if hm_list and hm_list[0] is not None else None
                print(f"  [dbg] 槽位 {name}: hms len={len(hm_list)} "
                      f"hms[0]={'None' if hm is None else str(np.asarray(hm).shape)}",
                      flush=True)
        # fused 热力图：各槽位 hms 均值（等权近似，grid 内）
        fused_hm = None
        for name, hm_list in hms.items():
            hm = hm_list[0] if hm_list and hm_list[0] is not None else None
            if hm is None:
                continue
            hm = np.asarray(hm, dtype=np.float32)
            agg.setdefault(name, {"hit": [], "energy": []})
            hr, en, _ = hit_rate_and_energy(hm, boxes)
            if hr is not None:
                agg[name]["hit"].append(hr)
                agg[name]["energy"].append(en)
            fused_hm = hm if fused_hm is None else fused_hm + hm
        if fused_hm is not None:
            hr, en, _ = hit_rate_and_energy(fused_hm, boxes)
            if hr is not None:
                fused_hit.append(hr)
                fused_energy.append(en)
        n_done += 1
        if n_done >= n:
            break
        if n_done % 10 == 0:
            print(f"  [loc] {n_done}/{n} 图", flush=True)

    print(f"\n=== 定位基线（test 缺陷图 n={n_done}，含框图 {n_box}，grid={cfg['backbone']['grid']}）===")
    print(f"{'槽位':<8} {'HitRate@top1%':>12} {'框内能量占比':>12} {'样本':>5}")
    for name, v in agg.items():
        if v["hit"]:
            print(f"{name:<8} {np.mean(v['hit']):>12.3f} {np.mean(v['energy']):>12.3f} "
                  f"{len(v['hit']):>5}")
    if fused_hit:
        print(f"{'fused':<8} {np.mean(fused_hit):>12.3f} {np.mean(fused_energy):>12.3f} "
              f"{len(fused_hit):>5}")


if __name__ == "__main__":
    main()
