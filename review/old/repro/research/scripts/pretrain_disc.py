"""跨品类预训练 disc 头（§9.1，U11）

用 MVTec 全部 14 品类 train/good 正常图 + 合成伪异常，预训练判别头。
关键：disc 头输入是 DINO patch 特征（384 维），与品类无关 → 预训练后可
跨品类迁移。验证方式：预训练头直接在 data_local gold_finger 上评分，
对比"每品类从头训练"的增益（若有）与迁移成本（省掉的训练时间）。

用法：
  python scripts/pretrain_disc.py                 # MVTec 预训练 + data_local 验证
"""
import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.backbone.dino import FrozenDINO
from src.slots.disc import DiscSlot, DiscHead, synth_pseudo
from src.data.datalocal import load_category

MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "transistor", "wood", "zipper"
]


def collect_mvtec_good(root, per_cat=40):
    """收集 MVTec 各品类 train/good 图路径（子采样 per_cat）"""
    paths = []
    for cat in MVTEC_CATEGORIES:
        d = os.path.join(root, cat, "train", "good")
        if not os.path.isdir(d):
            continue
        imgs = sorted(os.path.join(d, f) for f in os.listdir(d)
                      if f.lower().endswith((".png", ".jpg", ".jpeg")))
        rng = np.random.default_rng(42)
        imgs = list(rng.choice(imgs, min(per_cat, len(imgs)), replace=False))
        paths += imgs
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mvtec", default=r"D:\CGAIC\data_origin\mvtec")
    ap.add_argument("--per-cat", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "outputs", "m0", "disc_pretrain.pt"))
    args = ap.parse_args()

    import cv2
    from src.common.io import load_image

    device = "cuda"
    bb = FrozenDINO(output_layer=9, grid=32, device=device)

    # 1. 收集 MVTec 正常图
    good_paths = collect_mvtec_good(args.mvtec, args.per_cat)
    print(f"收集 MVTec 正常图 {len(good_paths)} 张（{len(MVTEC_CATEGORIES)} 品类 × ~{args.per_cat}）")

    # 2. 提特征（正常 + 伪异常），每图子采样 patch
    rng = np.random.default_rng(42)
    neg_all, pos_all = [], []
    t0 = time.time()
    for i, p in enumerate(good_paths):
        img = load_image(p)
        neg_f = bb.extract_tiles([img], batch=1).cpu()
        neg_all.append(neg_f.flatten(2).permute(0, 2, 1).reshape(-1, bb.dim))
        for _ in range(4):  # 每图 4 个伪异常
            aug, _ = synth_pseudo(img, rng)
            pos_f = bb.extract_tiles([aug], batch=1).cpu()
            pos_all.append(pos_f.flatten(2).permute(0, 2, 1).reshape(-1, bb.dim))
        if (i + 1) % 50 == 0:
            print(f"  提特征 {i+1}/{len(good_paths)} ({time.time()-t0:.0f}s)")
    neg = torch.cat(neg_all)
    pos = torch.cat(pos_all)
    cap = 80000
    neg = neg[torch.randperm(len(neg))[:cap]]
    pos = pos[torch.randperm(len(pos))[:cap]]
    print(f"neg={len(neg)} pos={len(pos)} patch，特征提取 {time.time()-t0:.0f}s")

    # 3. 训练预训练头
    head = DiscHead(bb.dim, 128).to(device)
    X = torch.cat([neg, pos]).float().to(device)
    y = torch.cat([torch.zeros(len(neg)), torch.ones(len(pos))]).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    w_pos = len(neg) / max(len(pos), 1)
    for ep in range(args.epochs):
        perm = torch.randperm(len(X))
        total = 0.0
        for i in range(0, len(X), 4096):
            xb, yb = X[perm[i:i + 4096]], y[perm[i:i + 4096]]
            logit = head(xb)
            loss = F.binary_cross_entropy_with_logits(
                logit, yb, pos_weight=torch.tensor(w_pos, device=device))
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        print(f"  epoch {ep+1}/{args.epochs} loss={total/(len(X)//4096+1):.4f}")
    head.eval()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(head.state_dict(), args.out)
    print(f"预训练头已保存: {args.out}")

    # 4. data_local gold_finger 验证：加载预训练头直接评分
    from src.fusion.calibrate import CDFCalibrator
    from sklearn.metrics import roc_auc_score
    bundle = load_category(r"D:\CGAIC\data_local", "gold_finger", 100, 30, 42)
    slot = DiscSlot({"hidden": 128}, bb, device)
    slot.head.load_state_dict(head.state_dict())
    slot.head.eval()

    # 直接评分（不重训，保留预训练权重）：score_tiles 需要 (T,384,G,G)
    def score_img(p):
        img = load_image(p)
        f = bb.extract_tiles([img]).cpu()
        ts, _ = slot.score_tiles(f)
        return float(ts.max())

    tr_scores = [score_img(p) for p in bundle["init_normal"][:30]]
    cal = CDFCalibrator(256).fit(tr_scores)
    labels = [y for _, y in bundle["test"][:60]]
    scores = [cal.transform([score_img(p)])[0] for p, _ in bundle["test"][:60]]
    print(f"[gold_finger] 预训练头直接迁移 AUROC={roc_auc_score(labels, scores):.4f}")


if __name__ == "__main__":
    main()
