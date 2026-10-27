"""域漂移诊断 + 域归一化验证（U97，2026-08-24，评审 §8"无域适配层"的对症攻坚）

评审 §8 核心批评：方案无"特征对齐/BN 统计适配/域归一化"层，跨域（gyudet/data_local）
靠被动在线学习，数据表现是架构必然结果。

本脚本诊断 gyudet train 域 → test 域的 DINO 特征漂移统计本质，并验证几种域归一化
方案能否提升跨域特征可分性：
  raw         原始特征 + cosine 距离（sem 现状）
  instnorm    实例归一化（每图自身 z-score，消除单图统计偏移）
  trainzscore 用 train 统计 z-score 归一化 test
  whitening   PCA 白化（train 拟合白化矩阵，消除相关性 + 方差归一）

用法: python scripts/diag_domain_shift.py
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
from src.data import gyudet
from sklearn.metrics import roc_auc_score

CFG = os.path.join(os.path.dirname(__file__), "..", "configs", "m0.yaml")


def extract_patches(backbone, paths, device, per_img=64):
    """提取每张图的 patch 特征（采样 per_img 个 patch），返回 (N,384) + 图索引。"""
    feats, gidx = [], []
    for gi, p in enumerate(paths):
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        with torch.no_grad():
            toks = backbone(x)  # (1,384,G,G)
        f = toks.flatten(2).squeeze(0).permute(1, 0).cpu().numpy()  # (G*G,384)
        sel = np.random.default_rng(gi).choice(len(f), min(per_img, len(f)), replace=False)
        feats.append(f[sel])
        gidx += [gi] * len(sel)
    return np.concatenate(feats), np.array(gidx)


def nn_distance_auroc(train_feats, test_norm_feats, test_def_feats):
    """patch 到 train 库的最近邻 cosine 距离 → AUROC（正常 vs 缺陷）。"""
    from sklearn.metrics.pairwise import cosine_similarity
    sims_n = cosine_similarity(test_norm_feats, train_feats).max(axis=1)
    sims_d = cosine_similarity(test_def_feats, train_feats).max(axis=1)
    labels = [0] * len(sims_n) + [1] * len(sims_d)
    scores = list(1 - sims_n) + list(1 - sims_d)
    return roc_auc_score(labels, scores)


def relative_distance_auroc(train_feats, test_norm_feats, test_def_feats,
                            test_norm_gidx, test_def_gidx):
    """相对距离 = 到库最近邻距离 - 图内最近邻距离（图内相对化，消除记忆库绝对距离偏差）。
    这是 U95 决策阈值失效（test/good fused 右移）的对症解 + 域适配方案。"""
    from sklearn.metrics.pairwise import cosine_similarity

    def rel(feats, gidx):
        d_lib = 1 - cosine_similarity(feats, train_feats).max(axis=1)  # 到库
        d_intra = np.zeros(len(feats))
        for gi in np.unique(gidx):
            m = gidx == gi
            intra = 1 - cosine_similarity(feats[m], feats[m])  # 图内成对
            np.fill_diagonal(intra, 1.0)
            d_intra[m] = intra.min(axis=1)
        return d_lib - d_intra

    s_n = rel(test_norm_feats, test_norm_gidx)
    s_d = rel(test_def_feats, test_def_gidx)
    labels = [0] * len(s_n) + [1] * len(s_d)
    return roc_auc_score(labels, list(s_n) + list(s_d))


def main():
    cfg = yaml.safe_load(open(CFG, encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = FrozenDINO(output_layer=9, grid=32, device=device)
    root = cfg["datasets"]["gyu_det"]

    tr_n, tr_d = gyudet._load_split(root, "train")
    te_n, te_d = gyudet._load_split(root, "test")
    rng = np.random.default_rng(42)
    tr_n = sorted(rng.choice(tr_n, min(60, len(tr_n)), replace=False).tolist())
    te_n = sorted(rng.choice(te_n, min(30, len(te_n)), replace=False).tolist())
    te_d = sorted(rng.choice(te_d, min(60, len(te_d)), replace=False).tolist())
    print(f"[U97] train正常 {len(tr_n)} / test正常 {len(te_n)} / test缺陷 {len(te_d)}", flush=True)

    tr_f, tr_g = extract_patches(backbone, tr_n, device)
    te_n_f, te_n_g = extract_patches(backbone, te_n, device)
    te_d_f, te_d_g = extract_patches(backbone, te_d, device)

    # 诊断：train vs test 通道统计偏移
    d_mu = np.abs(tr_f.mean(0) - te_n_f.mean(0)).mean()
    d_std = np.abs(tr_f.std(0) - te_n_f.std(0)).mean()
    print(f"[U97] train vs test 通道均值差 {d_mu:.4f} / 标准差差 {d_std:.4f}", flush=True)

    def instnorm(f, gidx):
        out = np.zeros_like(f)
        for gi in np.unique(gidx):
            m = gidx == gi
            out[m] = (f[m] - f[m].mean(0)) / (f[m].std(0) + 1e-6)
        return out

    results = {}
    # raw
    results["raw(cosine)"] = nn_distance_auroc(tr_f, te_n_f, te_d_f)
    # instnorm：每图 z-score
    results["instnorm"] = nn_distance_auroc(
        instnorm(tr_f, tr_g), instnorm(te_n_f, te_n_g), instnorm(te_d_f, te_d_g))
    # trainzscore：用 train 统计 z-score
    mu, std = tr_f.mean(0), tr_f.std(0) + 1e-6
    results["trainzscore"] = nn_distance_auroc(
        tr_f, (te_n_f - mu) / std, (te_d_f - mu) / std)
    # whitening：PCA 白化（train 拟合）
    from sklearn.decomposition import PCA
    pca = PCA(n_components=128, whiten=True, random_state=42).fit(tr_f)
    results["whitening(pca128)"] = nn_distance_auroc(
        pca.transform(tr_f), pca.transform(te_n_f), pca.transform(te_d_f))
    # 相对距离（图内相对化）
    results["relative(图内相对化)"] = relative_distance_auroc(
        tr_f, te_n_f, te_d_f, te_n_g, te_d_g)

    print("[U97] 跨域特征可分性（patch 最近邻距离 AUROC，正常 vs 缺陷）:", flush=True)
    for k, v in results.items():
        print(f"  {k:20s} AUROC={v:.4f}", flush=True)

    # ---- data_local gold_finger：修复图(train) vs 原始图(test) 的域差 ----
    from src.data import datalocal
    dl_root = cfg["datasets"]["data_local"]
    b = datalocal.load_category(dl_root, "gold_finger", 100, 30, 42, max_test=0)
    tr_n2 = b["init_normal"]
    te_n2 = [p for p, y in b["test"] if y == 0]
    te_d2 = [p for p, y in b["test"] if y == 1]
    tr_n2 = sorted(rng.choice(tr_n2, min(40, len(tr_n2)), replace=False).tolist())
    te_n2 = sorted(rng.choice(te_n2, min(15, len(te_n2)), replace=False).tolist())
    te_d2 = sorted(rng.choice(te_d2, min(40, len(te_d2)), replace=False).tolist())
    print(f"\n[U97] data_local/gold_finger: train正常 {len(tr_n2)} / test正常 {len(te_n2)} / test缺陷 {len(te_d2)}", flush=True)
    tr_f2, tr_g2 = extract_patches(backbone, tr_n2, device)
    te_n_f2, te_n_g2 = extract_patches(backbone, te_n2, device)
    te_d_f2, te_d_g2 = extract_patches(backbone, te_d2, device)
    d_mu2 = np.abs(tr_f2.mean(0) - te_n_f2.mean(0)).mean()
    d_std2 = np.abs(tr_f2.std(0) - te_n_f2.std(0)).mean()
    print(f"[U97] data_local train vs test 通道均值差 {d_mu2:.4f} / 标准差差 {d_std2:.4f}", flush=True)
    r2 = {}
    r2["raw(cosine)"] = nn_distance_auroc(tr_f2, te_n_f2, te_d_f2)
    r2["instnorm"] = nn_distance_auroc(instnorm(tr_f2, tr_g2), instnorm(te_n_f2, te_n_g2), instnorm(te_d_f2, te_d_g2))
    mu2, std2 = tr_f2.mean(0), tr_f2.std(0) + 1e-6
    r2["trainzscore"] = nn_distance_auroc(tr_f2, (te_n_f2 - mu2) / std2, (te_d_f2 - mu2) / std2)
    r2["relative(图内相对化)"] = relative_distance_auroc(
        tr_f2, te_n_f2, te_d_f2, te_n_g2, te_d_g2)
    print("[U97] data_local/gold_finger 跨域特征可分性:", flush=True)
    for k, v in r2.items():
        print(f"  {k:20s} AUROC={v:.4f}", flush=True)


if __name__ == "__main__":
    main()
