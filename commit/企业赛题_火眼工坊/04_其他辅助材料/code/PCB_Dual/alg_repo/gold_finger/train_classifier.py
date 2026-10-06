# -*- coding: utf-8 -*-
"""
在 extract_features.py 生成的特征数据集上训练缺陷分类器（随机森林）。

用法:
    python extract_features.py            # 先生成 features_dataset.npz
    python train_classifier.py            # 交叉验证评估 + 训练最终模型

输出:
    defect_classifier.joblib   sklearn 原始模型 (备档/复训用)
    defect_classifier.npz      纯 NumPy 格式 (流水线在线推理用，
                               加载毫秒级且不依赖 sklearn)
"""
import os
import sys

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# 同义标签归并：划伤与划痕是同一类缺陷的不同叫法
LABEL_MERGE = {"HuaShang": "HuaHen"}
# 样本数低于该值的类别无法有效训练，直接剔除（会打印警告）
MIN_SAMPLES_PER_CLASS = 5


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    data_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "features_dataset.npz")
    model_path = os.path.join(here, "defect_classifier.joblib")

    data = np.load(data_path, allow_pickle=False)
    X, y = data["X"], data["y"].astype(str)
    feature_names = data["feature_names"].astype(str)

    y = np.array([LABEL_MERGE.get(lbl, lbl) for lbl in y])

    classes, counts = np.unique(y, return_counts=True)
    keep_classes = classes[counts >= MIN_SAMPLES_PER_CLASS]
    dropped = [(c, n) for c, n in zip(classes, counts) if n < MIN_SAMPLES_PER_CLASS]
    for c, n in dropped:
        print(f"[!] 类别 {c} 仅 {n} 个样本，低于 {MIN_SAMPLES_PER_CLASS}，本次训练剔除")

    sel = np.isin(y, keep_classes)
    X, y = X[sel], y[sel]
    print(f"\n训练样本 {len(y)}，类别 {len(keep_classes)} 个:")
    for c in keep_classes:
        print(f"  {c:10s} {(y == c).sum()}")

    clf = RandomForestClassifier(
        # 实测 200 棵与 400 棵 CV 精度持平 (0.9074 vs 0.9067)，
        # 模型加载与在线预测耗时减半
        n_estimators=200,
        min_samples_leaf=2,
        class_weight="balanced",   # 类别不均衡 (389 vs 12)，按频率加权
        random_state=42,
        n_jobs=-1,
    )

    # ---- 5 折分层交叉验证评估 ----
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(clf, X, y, cv=cv, n_jobs=-1)

    print("\n===== 5 折交叉验证结果 =====")
    print(classification_report(y, y_pred, digits=3))
    print("混淆矩阵 (行=真实, 列=预测):")
    labels_sorted = sorted(keep_classes)
    cm = confusion_matrix(y, y_pred, labels=labels_sorted)
    header = " " * 10 + "".join(f"{c:>9s}" for c in labels_sorted)
    print(header)
    for c, row in zip(labels_sorted, cm):
        print(f"{c:10s}" + "".join(f"{v:9d}" for v in row))

    # ---- 用全部数据训练最终模型 ----
    clf.fit(X, y)

    print("\n特征重要性 Top 15:")
    order = np.argsort(clf.feature_importances_)[::-1][:15]
    for i in order:
        print(f"  {feature_names[i]:16s} {clf.feature_importances_[i]:.4f}")

    joblib.dump(
        {"model": clf, "feature_names": list(feature_names),
         "classes": list(clf.classes_), "label_merge": LABEL_MERGE},
        model_path,
    )
    print(f"\n[OK] sklearn 模型已保存: {model_path}")

    # 导出纯 NumPy 格式供流水线在线推理（加载毫秒级，不依赖 sklearn）
    from forest_infer import NumpyForest, export_forest
    npz_path = os.path.join(here, "defect_classifier.npz")
    export_forest(clf, npz_path)

    # 自检：NumPy 推理结果必须与 sklearn 一致
    nf = NumpyForest(npz_path)
    p_np = nf.predict_proba(X)
    p_sk = clf.predict_proba(X)
    assert list(nf.classes_) == [str(c) for c in clf.classes_]
    assert np.allclose(p_np, p_sk, atol=1e-5), "NumPy 森林与 sklearn 预测不一致!"
    print(f"[OK] NumPy 推理模型已导出并通过一致性自检: {npz_path}")


if __name__ == "__main__":
    main()
