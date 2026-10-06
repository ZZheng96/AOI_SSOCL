"""sem 对照诊断：同一划分下隔离变量

A) demo4 式：整图单前向 → 全局 mean-pool → 图像级 kNN（demo4 honest core）
B) 整图单前向 → patch 级 kNN top-k（粒度不变，打分换 patch）
C) tiles36 → patch 级 kNN top-k → tile top-3（demo5 M0 当前实现）
若 A 高而 C 低 → tile 粒度或聚合有 bug；若都低 → 协议/口径差异。
"""
import os
import sys
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.common.tiling import compute_tiles, extract_tiles
from src.common.io import load_image
from src.backbone.dino import FrozenDINO
from src.data import datalocal, gyudet


def patch_flat(feats):
    return F.normalize(feats.flatten(2).permute(0, 2, 1).reshape(-1, feats.shape[1]).float(), dim=-1)


def nn_dist_chunked(query, bank, chunk=4096):
    """1 - 逐 patch 最大余弦相似度（分块防 OOM）。query (N,384), bank (M,384)"""
    outs = []
    for i in range(0, len(query), chunk):
        sims = query[i:i + chunk] @ bank.T
        outs.append(1 - sims.max(dim=-1).values)
    return torch.cat(outs)


def knn_cos_scores(query, bank):
    """query (N,384) -> 1 - max cosine to bank"""
    return (1 - query @ bank.T).max(dim=-1).values  # placeholder, replaced below


def main():
    with open(os.path.join(os.path.dirname(__file__), "..", "configs", "m0.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = "cuda"
    bb = FrozenDINO(output_layer=9, grid=32, device=device)

    for ds_name, bundle, n_eval in (
        ("gyudet", gyudet.load_gyudet(cfg["datasets"]["gyu_det"], 100, 30, 42), 150),
        ("gold_finger", datalocal.load_category(cfg["datasets"]["data_local"], "gold_finger", 100, 30, 42), 40),
    ):
        print(f"\n===== {ds_name} =====")
        # 训练特征：整图 + tiles 各提一次
        tr_whole, tr_tiles = [], []
        for p in bundle["init_normal"]:
            img = load_image(p)
            tr_whole.append(bb.extract_tiles([img]).cpu())
            ti = extract_tiles(img, compute_tiles(img.shape[:2], "tiles36", 512, 448))
            tr_tiles.append(bb.extract_tiles(ti).cpu())
        # A: 图像级池化库
        bank_img = F.normalize(torch.cat(tr_whole).mean(dim=(2, 3)).float(), dim=-1).to(device)
        # B/C: patch 库（分别来自整图 patch / tile patch，子采样）
        bank_pw = patch_flat(torch.cat(tr_whole)).to(device)
        bank_pt = patch_flat(torch.cat(tr_tiles))
        bank_pt = bank_pt[torch.randperm(len(bank_pt))[:200_000]].to(device)

        eval_set = bundle["test"][:n_eval]
        labels = [y for _, y in eval_set]
        sA, sB, sC = [], [], []
        for p, y in eval_set:
            img = load_image(p)
            fw = bb.extract_tiles([img]).cpu()                      # (1,384,32,32)
            emb = F.normalize(fw.mean(dim=(2, 3)), dim=-1).to(device)
            sA.append(float((1 - emb @ bank_img.T).max()))          # 1-最近邻相似度
            pw = patch_flat(fw).to(device)
            d = nn_dist_chunked(pw, bank_pw)
            sB.append(float(d.topk(52).values.mean()))
            ti = extract_tiles(img, compute_tiles(img.shape[:2], "tiles36", 512, 448))
            ft = bb.extract_tiles(ti).cpu()
            pt = patch_flat(ft).to(device)
            dt = nn_dist_chunked(pt, bank_pt).reshape(len(ti), -1)
            ts = dt.topk(max(1, int(dt.shape[1] * 0.05)), dim=-1).values.mean(dim=-1)
            sC.append(float(ts.topk(min(3, len(ts))).values.mean()))
        for tag, s in (("A 整图池化kNN(demo4式)", sA), ("B 整图patch-kNN", sB), ("C tiles36(demo5)", sC)):
            print(f"  {tag}: AUROC={roc_auc_score(labels, s):.4f}")


if __name__ == "__main__":
    main()
