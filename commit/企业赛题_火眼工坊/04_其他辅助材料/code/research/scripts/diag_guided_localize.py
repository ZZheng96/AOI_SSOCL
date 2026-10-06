"""U77（2026-08-19）：两级结构粗筛定位诊断 -- 三种粗筛信号在 test 缺陷图上的定位召回率。

背景：U76 定论 demo5 诚实冲 0.9 需"整图粗筛定位 + 疑点 crop 精查"两级结构。
关键难点 = 粗筛信号能否在 test 域定位到缺陷（U70 已证无监督聚焦特征判别力≈随机，
但那是"特征判别"不是"定位召回"——定位只需框内 patch 高分）。

度量：对每个 GT 框，算"框内最高分 patch 的分数分位数"（rank percentile，
越小越靠前）；统计分位数 < p 的框占比 = top-p 定位召回率。
三种信号：
  A. 无监督：patch 到 train 正常 patch 均值向量的 L2 距离
  B. sem coreset：patch 到 train 正常 coreset 的最远点贪心库距离（近似 sem 槽）
  C. 对照：随机（理论召回 = p）

诚实边界：train 正常图建库（协议内），test 只用 GT 框做评价（只验不选）。
速度：DINO 提取 ~40 张（100 train 采样 30 + test 30）≈ 30s + 距离计算（CPU 可）。
"""
import os
import sys
import json
import time

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
import torch.nn.functional as F

from src.backbone.dino import FrozenDINO

GYU_ROOT = r"D:\CGAIC\data_origin\GYU-DET"
SEED = 42


def parse_yolo_boxes(lbl_path, img_w, img_h):
    boxes = []
    if not os.path.exists(lbl_path):
        return boxes
    with open(lbl_path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                _, cx, cy, w, h = (float(v) for v in parts[:5])
            except ValueError:
                continue
            if w <= 0 or h <= 0:
                continue
            boxes.append([(cx - w / 2) * img_w, (cy - h / 2) * img_h,
                          (cx + w / 2) * img_w, (cy + h / 2) * img_h])
    return boxes


def load_good_paths(split, n, seed=42):
    rng = np.random.default_rng(seed)
    img_dir = os.path.join(GYU_ROOT, split, "images")
    lbl_dir = os.path.join(GYU_ROOT, split, "labels")
    normal = []
    for f in sorted(os.listdir(img_dir)):
        if not f.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        lbl = os.path.join(lbl_dir, os.path.splitext(f)[0] + ".txt")
        if not (os.path.exists(lbl) and os.path.getsize(lbl) > 0):
            normal.append(os.path.join(img_dir, f))
    idx = rng.choice(len(normal), min(n, len(normal)), replace=False)
    return [normal[i] for i in idx]


def load_defect_paths(split, n, seed=42):
    rng = np.random.default_rng(seed)
    img_dir = os.path.join(GYU_ROOT, split, "images")
    lbl_dir = os.path.join(GYU_ROOT, split, "labels")
    defect = []
    for f in sorted(os.listdir(img_dir)):
        if not f.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        lbl = os.path.join(lbl_dir, os.path.splitext(f)[0] + ".txt")
        if os.path.exists(lbl) and os.path.getsize(lbl) > 0:
            defect.append(os.path.join(img_dir, f))
    idx = rng.choice(len(defect), min(n, len(defect)), replace=False)
    return [defect[i] for i in idx]


def main():
    import cv2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[U77 定位诊断] device={device}", flush=True)

    # 1. 数据：train 正常 40（建库）+ test 缺陷 30（定位评测）
    good_paths = load_good_paths("train", 40, SEED)
    def_paths = load_defect_paths("test", 30, SEED)
    print(f"  train/good={len(good_paths)} test/defect={len(def_paths)}", flush=True)

    # 2. backbone
    backbone = FrozenDINO(model_name="vits14", output_layer=9, grid=32, device=device)
    G = backbone.grid

    # 3. train 正常 patch 库（简化：均值向量 μ + 随机 512 coreset）
    t0 = time.time()
    good_feats = []
    for i, p in enumerate(good_paths):
        img = cv2.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        f = backbone.extract_tiles([img], batch=8)[0]     # (384,32,32)
        good_feats.append(f)
        if (i + 1) % 20 == 0:
            print(f"  提取 good {i + 1}/{len(good_paths)} ({time.time() - t0:.0f}s)", flush=True)
    good_feats = torch.stack(good_feats)                  # (N,384,32,32)
    flat = good_feats.flatten(2).permute(0, 2, 1).reshape(-1, good_feats.shape[1])  # (N*1024,384)
    flat = F.normalize(flat.float(), dim=1)
    mu = flat.mean(dim=0)                                 # 均值向量
    # 贪心 coreset 512（farthest-point，简版）
    rng = np.random.default_rng(SEED)
    idx = [int(rng.integers(len(flat)))]
    d = 1 - flat @ flat[idx[0]]
    for _ in range(511):
        i = int(torch.argmax(d).item())
        idx.append(i)
        d = torch.minimum(d, 1 - flat @ flat[i])
    coreset = flat[idx]                                   # (512,384)
    print(f"  库构建完成: 均值μ + coreset={coreset.shape} ({time.time() - t0:.0f}s)", flush=True)

    # 4. test 缺陷图定位评测
    n_boxes = 0
    rank_sig_a, rank_sig_b = [], []    # 框内最高分 patch 的分位数
    for i, p in enumerate(def_paths):
        lbl = os.path.join(os.path.dirname(os.path.dirname(p)), "labels",
                           os.path.splitext(os.path.basename(p))[0] + ".txt")
        img = cv2.imread(p)
        h, w = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = parse_yolo_boxes(lbl, w, h)
        if not boxes:
            continue
        f = backbone.extract_tiles([img], batch=8)[0]     # (384,32,32)
        flat_t = f.reshape(f.shape[0], -1).T               # (1024,384)
        flat_t = F.normalize(flat_t.float(), dim=1).cpu()
        # 信号 A：到均值 μ 的 L2 距离
        d_a = torch.norm(flat_t - mu.cpu(), dim=1)        # (1024,)
        # 信号 B：到 coreset 最近邻距离
        sims = (flat_t.cpu() @ coreset.cpu().T).max(dim=1).values
        d_b = 1 - sims                                    # (1024,)
        # GT 框 -> patch 网格坐标（32x32，patch 覆盖全图）
        for bx in boxes:
            cx = (bx[0] + bx[2]) / 2 / w * G
            cy = (bx[1] + bx[3]) / 2 / h * G
            pi, pj = int(min(max(cy, 0), G - 1)), int(min(max(cx, 0), G - 1))
            pi = max(0, min(pi, G - 1)); pj = max(0, min(pj, G - 1))
            # 框内最高分 patch（含框边界 ±1 patch 邻域）
            y0 = max(0, int(bx[1] / h * G) - 1); y1 = min(G, int(bx[3] / h * G) + 1)
            x0 = max(0, int(bx[0] / w * G) - 1); x1 = min(G, int(bx[2] / w * G) + 1)
            if y1 <= y0 or x1 <= x0:
                y0 = max(0, pi - 1); y1 = min(G, pi + 2)
                x0 = max(0, pj - 1); x1 = min(G, pj + 2)
            grid_a = d_a.reshape(G, G)
            grid_b = d_b.reshape(G, G)
            in_a = grid_a[y0:y1, x0:x1].max().item()
            in_b = grid_b[y0:y1, x0:x1].max().item()
            # 分位数 = 框内最高分在全局排序中的位置比例
            rank_a = (d_a > in_a).float().mean().item()   # 0=最高分, 1=最低分
            rank_b = (d_b > in_b).float().mean().item()
            rank_sig_a.append(rank_a)
            rank_sig_b.append(rank_b)
            n_boxes += 1
        if (i + 1) % 10 == 0:
            print(f"  评测缺陷图 {i + 1}/{len(def_paths)} (框={n_boxes}) ({time.time() - t0:.0f}s)", flush=True)

    # 5. 汇总
    rng = np.random.default_rng(SEED)
    rand = rng.random(n_boxes)
    print(f"\n=== U77 定位召回（框内最高分 patch 的分位数，越小越好）===")
    print(f"  框数: {n_boxes}")
    for name, ranks in [("A 无监督μ距离", rank_sig_a), ("B sem coreset", rank_sig_b),
                        ("C 随机(理论)", rand)]:
        print(f"  [{name}] 均值分位={np.mean(ranks):.3f} "
              f"中位分位={np.median(ranks):.3f}")
        for p in (0.05, 0.10, 0.20, 0.30):
            rec = float(np.mean(np.array(ranks) < p))
            print(f"    top-{p * 100:.0f}% 定位召回 = {rec:.3f}")

    out = {"n_boxes": n_boxes,
           "A_unsup": {"mean_rank": round(float(np.mean(rank_sig_a)), 3),
                        "recall_top10": round(float(np.mean(np.array(rank_sig_a) < 0.1)), 3)},
           "B_sem": {"mean_rank": round(float(np.mean(rank_sig_b)), 3),
                     "recall_top10": round(float(np.mean(np.array(rank_sig_b) < 0.1)), 3)}}
    out_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "m0",
                            "diag_guided_localize.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(out, open(out_path, "w"), ensure_ascii=False, indent=2)
    print(f"\n[save] {out_path}  耗时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
