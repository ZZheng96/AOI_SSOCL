"""跨域机制A验证：伪异常域增强训练能否提升跨域判别方向（2026-08-26，小样本快速迭代）

理论（跨域攻坚方案 v1）：U98 证明"init 数据训练的判别头在 test 域漂移"（判别方向
跨域漂移），但 shead 判别头有效（gyudet 学习后 0.8856）——差异 = shead 训练时见过
伪异常 + 判别方向。本脚本验证：**训练时对 train 域图做域增强（亮度/对比度/模糊/
平移）扩充，判别头能否学到"域不变判别方向"，从而提升 test 域（跨域）判别力**。

口径：图像级判别（与 shead 同思路：patch 统计特征 → 判别头），对比
  base  = 原始 train 特征训练
  aug   = 原始 + 4 域增强副本训练
  patch = patch 最近邻距离（U97 口径，对照——已知跨域无效，验证脚本口径一致性）
测试集小样本（用户规则：每批 ≤10 图随机取，快速迭代）。

用法: python scripts/mechA_crossdomain.py
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

CFG = os.path.join(os.path.dirname(__file__), "..", "configs", "m0.yaml")

# ---- 域增强（图级，训练侧）----
def domain_aug(img, rng):
    """生成 4 个域增强副本：亮度±/gamma/模糊。返回 list[ndarray]。"""
    outs = []
    b = int(rng.integers(-25, 26))
    if b != 0:
        outs.append(np.clip(img.astype(np.float32) + b, 0, 255).astype(np.uint8))
    g = rng.choice([0.8, 1.25])
    outs.append(np.clip((img.astype(np.float32) / 255.0) ** g * 255.0, 0, 255).astype(np.uint8))
    k = rng.choice([3, 5])
    outs.append(cv2.GaussianBlur(img, (k, k), 0))
    dx = int(rng.integers(-6, 7))
    M = np.float32([[1, 0, dx], [0, 1, 0]])
    outs.append(cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                               borderMode=cv2.BORDER_REPLICATE))
    return outs


def extract_imgs(backbone, paths, device, rng, n_aug=0):
    """提取每张图的 patch 特征，聚合为图像级统计特征（mean/std，shead 同思路）。
    返回 X (N, 768)；n_aug>0 时对每张图追加 n_aug 个增强副本的特征。"""
    X = []
    for p in paths:
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        imgs = [img] + (domain_aug(img, rng) if n_aug > 0 else [])
        for im in imgs:
            x = torch.from_numpy(im).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
            with torch.no_grad():
                toks = backbone(x)   # (1,384,G,G)
            f = toks.flatten(2).squeeze(0).cpu().numpy()  # (G*G,384)
            X.append(np.concatenate([f.mean(0), f.std(0)]))
    return np.array(X)


def main():
    cfg = yaml.safe_load(open(CFG, encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = FrozenDINO(output_layer=9, grid=32, device=device)
    rng = np.random.default_rng(42)

    # ---- data_local gold_finger（跨域最难点）小样本 ----
    b = datalocal.load_category(cfg["datasets"]["data_local"], "gold_finger",
                                100, 30, 42, max_test=0)
    tr_n = sorted(rng.choice(b["init_normal"], 30, replace=False).tolist())
    tr_d = sorted(rng.choice(b["init_defect"], 15, replace=False).tolist())
    te_n = sorted(rng.choice([p for p, y in b["test"] if y == 0], 5, replace=False).tolist())
    te_d = sorted(rng.choice([p for p, y in b["test"] if y == 1], 5, replace=False).tolist())
    print(f"[mechA] gold_finger 小样本: train正常 {len(tr_n)} train缺陷 {len(tr_d)} "
          f"test正常 {len(te_n)} test缺陷 {len(te_d)}", flush=True)

    # 图像级特征（shead 同思路：patch mean/std）
    X_tr_n = extract_imgs(backbone, tr_n, device, rng)
    X_tr_d = extract_imgs(backbone, tr_d, device, rng)
    X_te_n = extract_imgs(backbone, te_n, device, rng)
    X_te_d = extract_imgs(backbone, te_d, device, rng)

    from sklearn.linear_model import LogisticRegression
    def img_clf_auroc(Xn, Xd, te_n, te_d):
        X = np.concatenate([Xn, Xd])
        yy = np.array([0] * len(Xn) + [1] * len(Xd))
        clf = LogisticRegression(max_iter=500).fit(X, yy)
        s_n = clf.predict_proba(te_n)[:, 1]
        s_d = clf.predict_proba(te_d)[:, 1]
        return roc_auc_score([0] * len(s_n) + [1] * len(s_d), list(s_n) + list(s_d))

    # base：原始 train 训练
    auroc_base = img_clf_auroc(X_tr_n, X_tr_d, X_te_n, X_te_d)
    # aug：原始 + 4 增强副本
    X_tr_n_aug = extract_imgs(backbone, tr_n, device, rng, n_aug=4)
    X_tr_d_aug = extract_imgs(backbone, tr_d, device, rng, n_aug=4)
    auroc_aug = img_clf_auroc(X_tr_n_aug, X_tr_d_aug, X_te_n, X_te_d)

    # patch 距离口径（U97 对照）
    from sklearn.metrics.pairwise import cosine_similarity
    def patch_auroc():
        feats = {}
        for name, paths in [("trn", tr_n), ("ten", te_n), ("ted", te_d)]:
            fs = []
            for p in paths:
                img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
                x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
                with torch.no_grad():
                    toks = backbone(x)
                fs.append(toks.flatten(2).squeeze(0).permute(1, 0).cpu().numpy())
            feats[name] = np.concatenate(fs)
        s_n = 1 - cosine_similarity(feats["ten"], feats["trn"]).max(1)
        s_d = 1 - cosine_similarity(feats["ted"], feats["trn"]).max(1)
        return roc_auc_score([0] * len(s_n) + [1] * len(s_d), list(s_n) + list(s_d))
    auroc_patch = patch_auroc()

    print(f"\n[mechA] gold_finger 跨域判别 AUROC（test 10 图小样本）:", flush=True)
    print(f"  patch 距离（U97 口径，对照）: {auroc_patch:.4f}", flush=True)
    print(f"  图像级判别 base（原始 train）: {auroc_base:.4f}", flush=True)
    print(f"  图像级判别 aug（+4 域增强） : {auroc_aug:.4f}", flush=True)
    print(f"  增强增益: {auroc_aug - auroc_base:+.4f}", flush=True)


if __name__ == "__main__":
    main()
