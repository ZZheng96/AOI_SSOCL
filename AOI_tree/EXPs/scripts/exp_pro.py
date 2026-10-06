"""PRO 定位精度评测（U102，2026-08-24，评审 §13"定位精度未评测"可补项）

用 MVTec ground_truth 掩码评测 L2 像素级掩码的定位精度：
  - 对每个缺陷图 predict 得像素级 mask（L2 输出）
  - 读 GT 掩码（ground_truth/{defect_type}/{name}_mask.png）
  - 逐 GT 连通域算 overlap（预测掩码覆盖 GT 区域的比例），取平均 = PRO
诚实边界：这是简化 PRO（覆盖率口径），非标准 AUPRO 曲线（需阈值扫描 FPR-TPR）。

用法: python scripts/exp_pro.py [--category bottle] [--n 20]
"""
import os
import sys

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import argparse
import numpy as np
import cv2
import yaml
import torch

from m0_baseline import build
from src.eval.offline import Pipeline
from src.data import mvtec_like

CFG = os.path.join(os.path.dirname(__file__), "..", "configs", "m0.yaml")


def gt_mask_path(img_path):
    """test/{defect}/{name}.png -> ground_truth/{defect}/{name}_mask.png"""
    d = os.path.dirname(img_path)
    name = os.path.splitext(os.path.basename(img_path))[0]
    defect = os.path.basename(d)
    gt = os.path.join(os.path.dirname(d), "..", "ground_truth", defect, f"{name}_mask.png")
    return os.path.normpath(gt)


def per_region_overlap(pred_mask, gt_mask):
    """逐 GT 连通域 overlap（预测掩码覆盖 GT 区域的比例）平均。"""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(gt_mask, 8)
    overlaps = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 4:
            continue
        region_gt = labels[y:y + h, x:x + w] == i
        region_pred = pred_mask[y:y + h, x:x + w]
        overlap = region_pred[region_gt].mean()   # GT 区域内预测为正的比例
        overlaps.append(float(overlap))
    return np.mean(overlaps) if overlaps else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="bottle")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--config", default=CFG)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = cfg["datasets"]["mvtec"]
    b = mvtec_like.load_category(root, args.category, 100, 30, 42,
                                 n_eval_good=0, max_test=0)
    backbone, slots = build(cfg, device)
    pipe = Pipeline(cfg, backbone, slots)
    pipe.fit(b)

    defects = [p for p, y in b["test"] if y == 1]
    rng = np.random.default_rng(42)
    defects = sorted(rng.choice(defects, min(args.n, len(defects)), replace=False).tolist())

    pros, missing_gt = [], 0
    for p in defects:
        gtp = gt_mask_path(p)
        if not os.path.exists(gtp):
            missing_gt += 1
            continue
        gt = cv2.imread(gtp, cv2.IMREAD_GRAYSCALE)
        gt = (gt > 0).astype(np.uint8)
        r = pipe.predict(p)
        pred = r.get("mask")
        if pred is None:
            pros.append(0.0)
            continue
        pred = (np.asarray(pred) > 0).astype(np.uint8)
        # 尺寸对齐
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        pros.append(per_region_overlap(pred, gt))

    pro = float(np.mean(pros)) if pros else 0.0
    print(f"[U102] {args.category}: PRO={pro:.4f}（{len(pros)} 张缺陷，"
          f"{missing_gt} 张无 GT）", flush=True)


if __name__ == "__main__":
    main()
