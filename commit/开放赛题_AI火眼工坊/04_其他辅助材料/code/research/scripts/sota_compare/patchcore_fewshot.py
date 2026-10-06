# -*- coding: utf-8 -*-
"""PatchCore few-shot（100 正常）复测 —— SOTA 对比（S2）之一。

协议与 demo5 一致：train/good 随机抽 100 张（seed=42）建记忆库，
test 全量评估图像级 AUROC（图像分 = patch 最近邻距离最大值）。
实现忠实于 PatchCore 论文：WideResNet50_2 层1/2/3 特征（上采样对齐到
16x16 patch）+ greedy coreset 10% + kNN 距离。
"""
import os
import sys
import json
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from sklearn.metrics import roc_auc_score
from PIL import Image

sys.stdout.reconfigure(line_buffering=True)  # 进度实时可见

SEED = 42
N_INIT = 100          # 赛题协议：100 张正常
IMG_SIZE = 256
CORESET_RATIO = 0.10  # 标准 PatchCore coreset 采样比例


def set_seed():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def load_paths(root, limit=None):
    paths = sorted(os.listdir(root))
    if limit is not None:
        rng = np.random.RandomState(SEED)
        idx = rng.choice(len(paths), limit, replace=False)
        paths = [paths[i] for i in idx]
    return [os.path.join(root, p) for p in paths]


_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def extract_patch_features(model, paths, device, bs=16):
    """逐批提取，返回 (n, 256, D) 每图 patch 级特征（16x16=256 patch）。"""
    all_feats = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(paths), bs):
            batch_paths = paths[i:i + bs]
            batch = torch.stack([
                _transform(Image.open(p).convert("RGB")) for p in batch_paths
            ]).to(device)
            x = model.conv1(batch); x = model.bn1(x); x = model.relu(x)
            x = model.maxpool(x)
            f1 = model.layer1(x)   # (b, 64,  64, 64)
            f2 = model.layer2(f1)  # (b,128,  32, 32)
            f3 = model.layer3(f2)  # (b,256,  16, 16)
            # 上采样对齐到 16x16，通道维拼接
            f1 = F.interpolate(f1, size=(16, 16), mode="bilinear", align_corners=False)
            f2 = F.interpolate(f2, size=(16, 16), mode="bilinear", align_corners=False)
            cat = torch.cat([f1, f2, f3], dim=1)          # (b, 448, 16, 16)
            b, c, h, w = cat.shape
            cat = cat.permute(0, 2, 3, 1).reshape(b, h * w, c)  # (b, 256, 448)
            all_feats.append(cat.cpu())
    return torch.cat(all_feats, dim=0)  # (n, 256, 448)


def greedy_coreset(feat, ratio=CORESET_RATIO):
    """Greedy coreset（逐点最大最小距离）。feat: (n, d)。
    预计算全距离矩阵后迭代，避免逐点 cdist 的重复开销。
    """
    n = feat.shape[0]
    m = max(1, int(n * ratio))
    feat = feat.float()
    # 预计算成对距离矩阵（CPU 分块）
    D = torch.zeros(n, n)
    bs = 1024
    for i in range(0, n, bs):
        D[i:i + bs] = torch.cdist(feat[i:i + bs].cpu(), feat.cpu())
    idx = [0]
    dists = D[0].clone()
    for k in range(1, m):
        j = int(torch.argmax(dists))
        idx.append(j)
        torch.minimum(dists, D[j], out=dists)
        if k % 500 == 0:
            print(f"  coreset {k}/{m}")
    return feat[torch.tensor(idx)]


def knn_dist(q, bank, bs=1024):
    """q: (n_patch, d) -> 每 patch 最近邻距离。"""
    ds = []
    for i in range(0, q.shape[0], bs):
        d = torch.cdist(q[i:i + bs], bank).min(dim=1).values
        ds.append(d)
    return torch.cat(ds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mvtec", default=r"D:\CGAIC\data_origin\mvtec")
    ap.add_argument("--categories", nargs="+",
                    default=["bottle", "capsule", "cable", "transistor", "screw"])
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"[PatchCore few-shot N={N_INIT}] device={device}")
    set_seed()
    model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1).to(device)

    results = {}
    for cat in args.categories:
        print(f"\n===== {cat} =====")
        tr_paths = load_paths(os.path.join(args.mvtec, cat, "train", "good"),
                              limit=N_INIT)
        te_root = os.path.join(args.mvtec, cat, "test")
        print(f"  train: {len(tr_paths)} (few-shot {N_INIT})")

        tr_f = extract_patch_features(model, tr_paths, device)       # (100,256,448)
        tr_flat = tr_f.reshape(-1, tr_f.shape[-1])
        print(f"  train patches: {tr_flat.shape}")
        bank = greedy_coreset(tr_flat).to(device)
        print(f"  bank: {bank.shape}")

        ys, scores = [], []
        for sub in sorted(os.listdir(te_root)):
            sub_root = os.path.join(te_root, sub)
            if not os.path.isdir(sub_root):
                continue
            label = 0 if sub == "good" else 1
            te_paths = load_paths(sub_root)
            if not te_paths:
                continue
            te_f = extract_patch_features(model, te_paths, device)   # (m,256,448)
            img_scores = []
            for j in range(te_f.shape[0]):
                d = knn_dist(te_f[j].to(device), bank)               # (256,)
                img_scores.append(float(d.max()))                    # 图像级 = max patch
            scores.extend(img_scores)
            ys.extend([label] * len(te_paths))
            print(f"  test/{sub}: {len(te_paths)} imgs")
        auroc = roc_auc_score(ys, scores)
        results[cat] = round(auroc, 4)
        print(f"  ==> {cat} AUROC = {auroc:.4f}")

    print("\n===== 汇总 =====")
    for c, a in results.items():
        print(f"{c}: {a}")
    os.makedirs("outputs/sota", exist_ok=True)
    json.dump(results, open("outputs/sota/patchcore_fewshot100.json", "w"),
              ensure_ascii=False, indent=2)
    print("saved outputs/sota/patchcore_fewshot100.json")


if __name__ == "__main__":
    main()
