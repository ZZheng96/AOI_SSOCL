"""有监督缺陷类型判别器验证（U96，2026-08-24）

U93/U94 的归因失败根因：用了"无监督槽位签名 → 规则映射"，而槽位签名对五类无判别性。
正确思路：init_defect 缺陷图有类型标签（路径文件夹名 → 赛题五类），红线 3 允许用
init_defect 训练 → 有监督类型判别器。

本脚本验证：颜色（HSV）+ 几何（连通域）+ 纹理（DoG）特征 + 逻辑回归，能否区分五类。

用法: python scripts/exp_defect_classifier.py [--per-defect 8]
"""
import os
import sys
import argparse

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import cv2

from src.data import mvtec_like, datalocal

CFG = os.path.join(os.path.dirname(__file__), "..", "configs", "m0.yaml")

# 缺陷类型（文件夹名/类名）→ 赛题五类
FIVE_MAP = {
    # 尺寸偏差：几何形变
    "bent": "尺寸偏差", "bent_lead": "尺寸偏差", "bent_wire": "尺寸偏差",
    "squeeze": "尺寸偏差", "squeezed_teeth": "尺寸偏差", "fold": "尺寸偏差",
    "flip": "尺寸偏差", "rough": "尺寸偏差",
    # 缺件少件：缺失/剪断/断齿
    "missing_cable": "缺件少件", "missing_wire": "缺件少件",
    "broken_teeth": "缺件少件", "cut_lead": "缺件少件",
    "cut_inner_insulation": "缺件少件", "cut_outer_insulation": "缺件少件",
    "poke_insulation": "缺件少件",
    # 逻辑错误：顺序/错位
    "cable_swap": "逻辑错误", "misplaced": "逻辑错误",
    # 色彩变化
    "color": "色彩变化",
    # 其余 → 常见外观缺陷（默认）
}
_DATA_LOCAL = {"component": "缺件少件", "extra_part": "缺件少件",
               "gold_finger": "常见外观缺陷", "solder_smt": "常见外观缺陷"}
FIVE = ["尺寸偏差", "缺件少件", "逻辑错误", "色彩变化", "常见外观缺陷"]


def map_type(folder, category, dataset):
    if dataset == "data_local":
        return _DATA_LOCAL.get(category, "常见外观缺陷")
    return FIVE_MAP.get(folder, "常见外观缺陷")


def features(img, backbone=None, device="cpu"):
    """提取判别五类缺陷的特征向量：颜色（HSV 直方图+矩）+ 几何（连通域）+ 纹理（DoG + DINO）。"""
    h, w = img.shape[:2]
    # 缩放到工作分辨率（加速 + 尺度归一）
    if max(h, w) > 1024:
        s = 1024 / max(h, w)
        img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))))
    # 1. 颜色：HSV H(18)+S(16) 直方图 + H/S/V 矩（mean/std）
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [18], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    color = np.concatenate([h_hist, s_hist])
    color = color / (color.sum() + 1e-6)
    hm, sm, vm = hsv[:, :, 0].mean(), hsv[:, :, 1].mean(), hsv[:, :, 2].mean()
    hs, ss, vs = hsv[:, :, 0].std(), hsv[:, :, 1].std(), hsv[:, :, 2].std()
    color_mom = np.array([hm, sm, vm, hs, ss, vs], dtype=np.float32) / 255.0
    # 2. 几何：连通域数量 + 面积统计（归一化）
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, _, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    areas = stats[1:, 4].astype(np.float32)
    if len(areas) == 0:
        areas = np.array([0.0], dtype=np.float32)
    total = float(bw.shape[0] * bw.shape[1])
    geom = np.array([len(areas), areas.mean() / total,
                     areas.max() / total, areas.sum() / total], dtype=np.float32)
    # 3. 纹理：DoG 响应统计（双尺度带）
    g = gray.astype(np.float32) / 255.0
    dog = np.maximum(
        np.abs(cv2.GaussianBlur(g, (0, 0), 1) - cv2.GaussianBlur(g, (0, 0), 2)),
        np.abs(cv2.GaussianBlur(g, (0, 0), 2) - cv2.GaussianBlur(g, (0, 0), 4)))
    text = np.array([float(dog.mean()), float(dog.max())], dtype=np.float32)
    # 4. DINO 全局特征：patch token 空间均值（384 维，判别外观缺陷强）
    dino_feat = np.zeros(0, dtype=np.float32)
    if backbone is not None:
        import torch
        x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        with torch.no_grad():
            toks = backbone(x)  # (1,384,G,G)
        dino_feat = toks.mean(dim=(2, 3)).squeeze(0).float().cpu().numpy()
    return np.concatenate([color, color_mom, geom, text, dino_feat]).astype(np.float32)


def collect(root, per_defect):
    """收集 MVTec 全部 init_defect 类型标签 + 路径。"""
    paths, types = [], []
    for cat in mvtec_like.MVTEC_CATEGORIES:
        test_dir = os.path.join(root, cat, "test")
        for sub in sorted(os.listdir(test_dir)):
            if sub == "good" or not os.path.isdir(os.path.join(test_dir, sub)):
                continue
            for f in sorted(os.listdir(os.path.join(test_dir, sub)))[:per_defect]:
                if f.lower().endswith((".png", ".jpg")):
                    paths.append(os.path.join(test_dir, sub, f))
                    types.append(map_type(sub, cat, "mvtec"))
    return paths, types


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-defect", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    root = os.path.join("D:", os.sep, "CGAIC", "data_origin", "mvtec")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from src.backbone.dino import FrozenDINO
    backbone = FrozenDINO(output_layer=9, grid=32, device=device)

    paths, types = collect(root, args.per_defect)
    rng = np.random.default_rng(args.seed)
    # 分层切分 train/test（按类型），模拟 init_defect(训练) vs eval_defect(测试)
    from collections import defaultdict
    idx_by_type = defaultdict(list)
    for i, t in enumerate(types):
        idx_by_type[t].append(i)
    train_idx, test_idx = [], []
    for t, idxs in idx_by_type.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        n_tr = max(1, len(idxs) // 2)
        train_idx += idxs[:n_tr]
        test_idx += idxs[n_tr:]

    X = np.stack([features(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB), backbone, device)
                  for p in paths])
    y = np.array([FIVE.index(t) for t in types])

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    clf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                 random_state=args.seed, n_jobs=-1)
    clf.fit(X[train_idx], y[train_idx])
    pred = clf.predict(X[test_idx])
    acc = accuracy_score(y[test_idx], pred)
    print(f"[U96] 训练 {len(train_idx)} / 测试 {len(test_idx)}，类型分布: "
          f"{ {t: sum(1 for tt in types if tt==t) for t in FIVE} }", flush=True)
    print(f"[U96] 5 类归因 top-1 准确率 = {acc:.3f} ({sum(pred==y[test_idx])}/{len(test_idx)})",
          flush=True)
    print(classification_report(y[test_idx], pred, target_names=FIVE, zero_division=0),
          flush=True)


if __name__ == "__main__":
    main()
