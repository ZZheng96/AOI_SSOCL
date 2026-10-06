"""U78（2026-08-19）：patch 级判别可行性诊断 -- 框内 vs 框外 patch 特征可分性

速度红线矛盾：crop/block/guided 多前向方案（DINO 每批 ~400ms）全超 1s；
single 模式 1 次前向（~450ms）达标但 image 精度仅 0.7713。
破局候选：patch 级判别 = single 整图 1 次前向（管线已有）+ patch 级 MLP
（零额外 DINO）。可行性前提：框内 patch 特征 vs 框外 patch 特征可分。

度量：train/init_defect 图框内 patch（pos）vs 同图框外 patch（neg）做
kNN/线性判别，评估可分性（AUROC）；再测 test 图框内 patch 是否高分
（跨域判别）。诚实边界：train 域建库/训练，test 只评价。
"""
import os
import sys
import json

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
import torch.nn.functional as F
import cv2

from src.backbone.dino import FrozenDINO
from src.slots.shead import _parse_yolo_boxes

GYU_ROOT = r"D:\CGAIC\data_origin\GYU-DET"
SEED = 42


def load_paths(split, want_defect, n, seed=42):
    rng = np.random.default_rng(seed)
    img_dir = os.path.join(GYU_ROOT, split, "images")
    lbl_dir = os.path.join(GYU_ROOT, split, "labels")
    out = []
    for f in sorted(os.listdir(img_dir)):
        if not f.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        lbl = os.path.join(lbl_dir, os.path.splitext(f)[0] + ".txt")
        is_def = os.path.exists(lbl) and os.path.getsize(lbl) > 0
        if is_def == want_defect:
            out.append(os.path.join(img_dir, f))
    idx = rng.choice(len(out), min(n, len(out)), replace=False)
    return [out[i] for i in idx]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[U78 patch 级判别诊断] device={device}", flush=True)
    tr_def = load_paths("train", True, 100, SEED)   # init_defect（协议内）
    te_def = load_paths("test", True, 30, SEED)     # test 缺陷（只评价）
    backbone = FrozenDINO(model_name="vits14", output_layer=9, grid=32, device=device)
    G = backbone.grid

    def patch_feats(paths, boxes_fn=None):
        """提取整图 patch 特征 + 框内/框外标签。返回 (feats, labels, n_patch_img)。"""
        feats, labels = [], []
        n_patch = []
        for p in paths:
            img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            lbl = os.path.join(os.path.dirname(os.path.dirname(p)), "labels",
                               os.path.splitext(os.path.basename(p))[0] + ".txt")
            boxes = _parse_yolo_boxes(lbl, w, h)
            f = backbone.extract_tiles([img], batch=8)[0].cpu()  # (384,32,32)
            flat = F.normalize(f.reshape(f.shape[0], -1).T.float(), dim=1)  # (1024,384)
            lab = np.zeros((G, G), dtype=np.int64)
            for bx in boxes:
                y0 = int(bx[1] / h * G); y1 = max(y0 + 1, min(G, int(bx[3] / h * G)))
                x0 = int(bx[0] / w * G); x1 = max(x0 + 1, min(G, int(bx[2] / w * G)))
                lab[y0:y1, x0:x1] = 1
            lab = lab.reshape(-1)
            feats.append(flat)
            labels.append(lab)
            n_patch.append(G * G)
        return torch.cat(feats, dim=0), np.concatenate(labels), n_patch

    # 1. 训练集框内 vs 框外可分性（kNN 距离 AUROC）
    tr_f, tr_lab, _ = patch_feats(tr_def)
    pos = tr_f[tr_lab == 1]
    neg = tr_f[tr_lab == 0]
    print(f"  训练集: pos_patch={len(pos)} neg_patch={len(neg)}", flush=True)
    # 用 pos 建库（均值），测 neg 距离分布 vs pos 距离分布（留一近似：pos 前 80% 建库）
    n_pos = len(pos)
    k_pos = max(1, int(n_pos * 0.8))
    pos_bank = pos[:k_pos]
    d_pos = (1 - (pos[k_pos:] @ pos_bank.T)).min(dim=1).values
    d_neg = (1 - (neg[:k_pos] @ pos_bank.T)).min(dim=1).values
    from sklearn.metrics import roc_auc_score
    n_s = min(len(d_pos), len(d_neg))
    auc_in = float(roc_auc_score([0] * n_s + [1] * n_s,
                                 np.concatenate([d_neg[:n_s], d_pos[:n_s]])))
    print(f"  训练集 pos-vs-neg kNN 距离 AUROC = {auc_in:.4f}", flush=True)

    # 2. 跨域：test 缺陷图框内 patch vs 训练 neg 库距离（框内应更远=更高异常）
    te_f, te_lab, _ = patch_feats(te_def)
    tr_neg_bank = F.normalize(neg[:512].float(), dim=1)
    te_d = (1 - (te_f @ tr_neg_bank.T)).min(dim=1).values.numpy()
    te_pos_d = te_d[te_lab == 1]
    te_neg_d = te_d[te_lab == 0]
    print(f"  test 框内 patch 数={len(te_pos_d)} 框外={len(te_neg_d)}")
    print(f"  test 框内距离 mean={te_pos_d.mean():.4f} vs 框外 mean={te_neg_d.mean():.4f}")
    n_s = min(len(te_pos_d), len(te_neg_d))
    auc_cross = float(roc_auc_score([0] * n_s + [1] * n_s,
                                    np.concatenate([te_neg_d[:n_s], te_pos_d[:n_s]])))
    print(f"  test 跨域 框内-vs-框外 AUROC = {auc_cross:.4f}")

    # 3. 跨域 patch 级判别头（线性 LR 训练 -> test 判别）
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    # 子采样训练
    rng = np.random.default_rng(SEED)
    n_tr = min(len(pos), len(neg))
    i_pos = rng.choice(len(pos), n_tr, replace=False)
    i_neg = rng.choice(len(neg), n_tr, replace=False)
    X = np.vstack([pos[i_pos].numpy(), neg[i_neg].numpy()])
    y = np.array([1] * n_tr + [0] * n_tr)
    lr = Pipeline([("sc", StandardScaler()),
                   ("lr", LogisticRegression(C=0.1, max_iter=2000))])
    lr.fit(X, y)
    # test 上打分
    s_pos = lr.predict_proba(te_pos_d * 0 + te_f[te_lab == 1].numpy())[:, 1] if False else None
    X_te = te_f.numpy()
    s_all = lr.predict_proba(X_te)[:, 1]
    auc_lr = float(roc_auc_score(te_lab, s_all))
    print(f"  test LR patch 判别头 AUROC = {auc_lr:.4f}")

    # 4. 图像级：patch top-k 均值聚合的整图 AUROC（test 30 缺陷 + 30 正常）
    from scripts.diag_guided_localize import load_good_paths
    te_nor_paths = load_good_paths("test", 30, SEED)
    te_all_paths = te_def + te_nor_paths
    y_img = np.array([1] * len(te_def) + [0] * len(te_nor_paths))
    img_scores = []
    for p in te_all_paths:
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        f = backbone.extract_tiles([img], batch=8)[0].cpu()
        flat = F.normalize(f.reshape(f.shape[0], -1).T.float(), dim=1)
        s = lr.predict_proba(flat.numpy())[:, 1]
        topk = np.sort(s)[-5:].mean()
        img_scores.append(topk)
    auc_img = float(roc_auc_score(y_img, img_scores))
    print(f"  test 整图 top5 patch 聚合 AUROC = {auc_img:.4f}")

    out = {"auc_train_knn": round(auc_in, 4), "auc_cross_patch": round(auc_cross, 4),
           "auc_lr_patch": round(auc_lr, 4), "auc_img_top5": round(auc_img, 4)}
    out_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "m0",
                            "diag_patch_disc.json")
    json.dump(out, open(out_path, "w"), ensure_ascii=False, indent=2)
    print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
