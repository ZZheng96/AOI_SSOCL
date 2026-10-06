"""全局类型判别器预训练（U96，2026-08-24）

per-category 单独训练（init_defect ~30 张）样本太少：color 类型可能抽不到 + 类别
不平衡，导致覆盖矩阵里色彩归因 0、外观归因 0.301。正确架构：类型判别器是"缺陷
类型 → 特征"的通用映射，应跨品类聚合训练一次（所有 MVTec + data_local 品类的
init_defect），保存为全局模型，Pipeline.fit 时加载。

用法: python scripts/train_defect_classifier.py [--per-defect 8] [--out outputs/m0/defect_clf.pkl]
"""
import os
import sys
import argparse
import pickle

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import cv2
import yaml
import torch

from src.backbone.dino import FrozenDINO
from src.data import mvtec_like, datalocal
from src.decision.defect_classifier import (DefectClassifier, extract_features,
                                            type_from_path, FIVE)

CFG = os.path.join(os.path.dirname(__file__), "..", "configs", "m0.yaml")


def collect_mvtec(root, category, per_defect, n_init_defect):
    """收集某品类的 init_defect 路径（用与 load_category 相同的 seed 抽样）。"""
    test_dir = os.path.join(root, category, "test")
    defects = []
    for sub in sorted(os.listdir(test_dir)):
        if sub == "good" or not os.path.isdir(os.path.join(test_dir, sub)):
            continue
        defects += [os.path.join(test_dir, sub, f) for f in os.listdir(os.path.join(test_dir, sub))
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    rng = np.random.default_rng(42)
    n_init = min(n_init_defect, max(1, len(defects) // 2))
    sel = rng.choice(len(defects), n_init, replace=False) if n_init else []
    return sorted(defects[i] for i in sel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-defect", type=int, default=8)
    ap.add_argument("--n-init-defect", type=int, default=30)
    ap.add_argument("--out", default=os.path.join("outputs", "m0", "defect_clf.pkl"))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(CFG, encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = FrozenDINO(output_layer=9, grid=32, device=device)

    paths = []
    for cat in mvtec_like.MVTEC_CATEGORIES:
        paths += collect_mvtec(cfg["datasets"]["mvtec"], cat, args.per_defect,
                               args.n_init_defect)
    # data_local：4 品类 init_defect（datalocal.load_category 同 seed）
    dl_root = cfg["datasets"]["data_local"]
    for cat in datalocal.CATEGORIES:
        b = datalocal.load_category(dl_root, cat, 100, args.n_init_defect, 42, max_test=0)
        paths += b["init_defect"]

    types = [type_from_path(p) for p in paths]
    from collections import Counter
    print(f"[U96] 收集 {len(paths)} 张 init_defect，类型分布: {dict(Counter(types))}",
          flush=True)

    # 提取特征（DINO + 颜色 + 几何 + DoG）
    X, y = [], []
    skip = 0
    for i, p in enumerate(paths):
        try:
            img = cv2.imread(p)
            if img is None:
                print(f"  [U96] 跳过（imread None）: {p}", flush=True)
                skip += 1
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
            with torch.no_grad():
                feats = backbone(x)  # (1,384,G,G)
            X.append(extract_features(img, feats))
            y.append(FIVE.index(types[i]))
        except Exception as e:
            print(f"  [U96] 跳过（异常 {type(e).__name__}: {e}）: {p}", flush=True)
            skip += 1
        if (i + 1) % 50 == 0:
            print(f"  [U96] 特征提取 {i + 1}/{len(paths)}", flush=True)
    X = np.stack(X)
    y = np.array(y)
    print(f"[U96] 有效样本 {len(X)}（跳过 {skip}）", flush=True)

    clf = DefectClassifier(seed=42)
    from sklearn.ensemble import RandomForestClassifier
    clf.clf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                     random_state=42, n_jobs=-1)
    clf.clf.fit(X, y)
    clf.fitted = True
    clf._labels = sorted(set(types))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(clf, f)
    print(f"[U96] 全局类型判别器保存 {args.out}（{len(paths)} 训练样本）", flush=True)


if __name__ == "__main__":
    main()
