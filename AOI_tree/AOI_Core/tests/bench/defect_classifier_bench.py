"""五类缺陷归因协议化实测（生产模块路径，2026-10-11）。

镜像 EXPs exp_defect_classifier（U96）协议：MVTec 全 15 品类 test 缺陷目录
per_defect=8 抽样、seed=42 按类型分层对半切分、RandomForest(300, balanced)。
特征提取与主干全部用生产模块：algo.decision.defect_classifier.extract_features
（图像侧 46 维 + DINO 384 维）+ algo.backbone.dino.FrozenDINO。
对照文档 §3.2.8 五类归因 top-1：0.787/0.857/0.562/0.750/0.836。

用法：
    python -m tests.bench.defect_classifier_bench --out <json>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
EXPS_SRC = REPO_ROOT.parent / "EXPs" / "src"   # 协议品类清单（只读）
if str(EXPS_SRC) not in sys.path:
    sys.path.append(str(EXPS_SRC))

MV_ROOT = os.environ.get("AOI_DATA_MVTEC", r"D:\CGAIC\data_origin\mvtec")
FIVE = ["尺寸偏差", "缺件少件", "逻辑错误", "色彩变化", "常见外观缺陷"]


def collect(per_defect: int):
    from data import mvtec_like  # noqa: E402  EXPs 协议（只读）：品类清单
    from algo.decision.defect_classifier import type_from_path  # noqa: E402  生产模块
    paths, types = [], []
    for cat in mvtec_like.MVTEC_CATEGORIES:
        test_dir = os.path.join(MV_ROOT, cat, "test")
        for sub in sorted(os.listdir(test_dir)):
            d = os.path.join(test_dir, sub)
            if sub == "good" or not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d))[:per_defect]:
                if f.lower().endswith((".png", ".jpg")):
                    paths.append(os.path.join(d, f))
                    types.append(type_from_path(paths[-1]))
    return paths, types


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-defect", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    from algo.decision.defect_classifier import extract_features, FIVE as PROD_FIVE  # noqa: E402
    assert PROD_FIVE == FIVE, f"生产模块五类口径变化: {PROD_FIVE}"
    from algo.backbone.dino import FrozenDINO  # noqa: E402
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = FrozenDINO(output_layer=9, grid=32, device=device)

    paths, types = collect(args.per_defect)
    print(f"[data] {len(paths)} 张（per_defect={args.per_defect}），"
          f"分布: { {t: types.count(t) for t in FIVE} }", flush=True)

    rng = np.random.default_rng(args.seed)
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

    def feats_of(p):
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        if max(img.shape[:2]) > 1024:
            s = 1024 / max(img.shape[:2])
            img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)))
        x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        with torch.no_grad():
            f = backbone(x)
        return extract_features(img, f)

    X = np.stack([feats_of(p) for p in paths])
    y = np.array([FIVE.index(t) for t in types])

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    clf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                 random_state=args.seed, n_jobs=-1)
    clf.fit(X[train_idx], y[train_idx])
    pred = clf.predict(X[test_idx])
    acc = accuracy_score(y[test_idx], pred)
    rep = classification_report(y[test_idx], pred, target_names=FIVE,
                                zero_division=0, output_dict=True)
    per_class = {t: round(rep[t]["recall"], 3) for t in FIVE}
    print(f"[result] 训练 {len(train_idx)} / 测试 {len(test_idx)}；"
          f"top-1={acc:.3f}；分类 recall={per_class}", flush=True)

    out = {"protocol": "U96 exp_defect_classifier 镜像（生产模块特征+主干）",
           "per_defect": args.per_defect, "seed": args.seed,
           "n_train": len(train_idx), "n_test": len(test_idx),
           "top1": round(float(acc), 4), "per_class_recall": per_class,
           "doc_ref": {"尺寸偏差": 0.787, "缺件少件": 0.857, "逻辑错误": 0.562,
                       "色彩变化": 0.750, "常见外观缺陷": 0.836},
           "elapsed_s": round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[done] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
