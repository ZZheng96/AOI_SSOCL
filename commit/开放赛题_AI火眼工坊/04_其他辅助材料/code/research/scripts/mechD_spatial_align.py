"""跨域机制D验证：空间对齐 patch 记忆库（AnomalyDINO/DinoPatch 启发，2026-08-26，小样本）

理论：demo4 差分 0.9983（作弊配对）证明"同位置对比"对 gold_finger 极强；demo5 sem
全局最近邻（无空间位置）在跨域失效——局部缺陷 patch 被"任意位置相似 patch"淹没
（思路.txt 第 30 行根因）。DinoPatch/AnomalyDINO 证明"空间对齐记忆库"（每个空间
位置独立建正常 patch 记忆，同一位置才匹配）是标准解法。本机制 = 合法同位置对比
（train 正常图建空间记忆，无 test 配对，红线内）。

验证：train 正常图（N 张）按 (x,y) 位置聚合 patch 特征建空间记忆；test 每 patch
只与同位置的正常 patch 最近邻距离；图像分数 = 距离 top-k 均值；跨域 AUROC
（正常 vs 缺陷）。对比 U97 全局 patch 距离（gold raw 0.293 反向）。

用法: python scripts/mechD_spatial_align.py
"""
import os
import sys

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import cv2
import yaml
import torch

from src.backbone.dino import FrozenDINO
from src.data import datalocal
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

CFG = os.path.join(os.path.dirname(__file__), "..", "configs", "m0.yaml")


def extract_patches(backbone, paths, device, per_img=None):
    """提取每张图全 patch 特征 (G*G, 384)。per_img=None 全取。"""
    feats, gidx = [], []
    for gi, p in enumerate(paths):
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        with torch.no_grad():
            toks = backbone(x)
        f = toks.flatten(2).squeeze(0).permute(1, 0).cpu().numpy()   # (G*G, 384)
        if per_img is not None and f.shape[0] > per_img:
            sel = np.random.default_rng(gi).choice(f.shape[0], per_img, replace=False)
            f = f[sel]
        feats.append(f)
        gidx += [gi] * len(f)
    return feats, np.array(gidx)


def global_nn_auroc(tr_feats, te_n_feats, te_d_feats):
    """全局最近邻（U97 口径，对照）：test patch 到 train 全部 patch 的最近邻距离。"""
    tr_all = np.concatenate(tr_feats)
    s_n = np.concatenate([1 - cosine_similarity(f, tr_all).max(1) for f in te_n_feats])
    s_d = np.concatenate([1 - cosine_similarity(f, tr_all).max(1) for f in te_d_feats])
    return roc_auc_score([0] * len(s_n) + [1] * len(s_d), list(s_n) + list(s_d))


def spatial_align_auroc(tr_feats, G, te_n_feats, te_d_feats, topk_ratio=0.05):
    """空间对齐记忆库：每位置 (G,G) 聚合 train patch → test patch 只与同位置最近邻。
    图像分数 = 距离 top-k 均值。"""
    # 建空间记忆：每位置存 train patch（N 张该位置）
    pos_mem = {i: [] for i in range(G * G)}
    for f in tr_feats:
        for i in range(min(G * G, f.shape[0])):
            pos_mem[i].append(f[i])
    pos_mem = {i: np.stack(v) for i, v in pos_mem.items() if v}

    def img_scores(feats):
        out = []
        for f in feats:
            d = []
            for i in range(min(G * G, f.shape[0])):
                if i not in pos_mem:
                    continue
                d.append(1 - cosine_similarity(f[i][None], pos_mem[i]).max())
            d = np.array(d)
            k = max(1, int(len(d) * topk_ratio))
            out.append(float(np.partition(d, -k)[-k:].mean()))
        return np.array(out)

    s_n = img_scores(te_n_feats)
    s_d = img_scores(te_d_feats)
    return roc_auc_score([0] * len(s_n) + [1] * len(s_d), list(s_n) + list(s_d))


def main():
    cfg = yaml.safe_load(open(CFG, encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = FrozenDINO(output_layer=9, grid=32, device=device)
    G = 32
    rng = np.random.default_rng(42)

    for cat in ["gold_finger", "solder_smt"]:
        b = datalocal.load_category(cfg["datasets"]["data_local"], cat,
                                    100, 30, 42, max_test=0)
        tr_n = sorted(rng.choice(b["init_normal"], 20, replace=False).tolist())
        te_n = sorted(rng.choice([p for p, y in b["test"] if y == 0], 5, replace=False).tolist())
        te_d = sorted(rng.choice([p for p, y in b["test"] if y == 1], 5, replace=False).tolist())
        print(f"\n[mechD] {cat}: train正常 {len(tr_n)} test正常 {len(te_n)} "
              f"test缺陷 {len(te_d)}", flush=True)
        tr_f, _ = extract_patches(backbone, tr_n, device)
        ten_f, _ = extract_patches(backbone, te_n, device)
        ted_f, _ = extract_patches(backbone, te_d, device)
        au_global = global_nn_auroc(tr_f, ten_f, ted_f)
        au_sp = spatial_align_auroc(tr_f, G, ten_f, ted_f)
        print(f"  全局最近邻 AUROC（U97 口径对照）: {au_global:.4f}", flush=True)
        print(f"  空间对齐记忆库 AUROC（机制D）  : {au_sp:.4f}", flush=True)


if __name__ == "__main__":
    main()
