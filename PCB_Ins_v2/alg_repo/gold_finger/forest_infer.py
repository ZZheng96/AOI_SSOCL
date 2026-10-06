# -*- coding: utf-8 -*-
"""
随机森林纯 NumPy 推理器。

检测端加载 sklearn + joblib 反序列化模型要约 1 秒（还会刷大量
DeprecationWarning），而随机森林预测本质只是"按阈值走树分支"。
train_classifier.py 训练后把所有树导出为扁平数组 (defect_classifier.npz)，
这里用 NumPy 直接遍历，加载毫秒级，预测结果与 sklearn 完全一致，
检测端不再依赖 scikit-learn。
"""
import numpy as np


class NumpyForest:
    """加载 train_classifier.py 导出的 .npz 森林并做批量概率预测。

    数组布局: 所有树的节点拼接在一起，children 索引已加上各树偏移，
    roots 记录每棵树根节点的全局下标；叶子节点 feature = -2。
    """

    def __init__(self, path):
        d = np.load(path)
        self.feature = d["feature"]        # (总节点数,) int32
        self.threshold = d["threshold"]    # (总节点数,) float32
        self.left = d["left"]              # (总节点数,) int32 全局下标
        self.right = d["right"]
        self.leaf_proba = d["leaf_proba"]  # (总节点数, 类别数) 叶子的类别分布
        self.roots = d["roots"]            # (树数,) int32
        self.classes_ = d["classes"].astype(str)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.ascontiguousarray(X, dtype=np.float32)
        n = X.shape[0]
        # idx: (树数, 样本数)，所有树、所有样本同步下行，直到全部到达叶子
        idx = np.repeat(self.roots[:, None], n, axis=1)
        col = np.arange(n)[None, :]
        while True:
            f = self.feature[idx]
            at_leaf = f < 0
            if at_leaf.all():
                break
            xv = X[col, np.maximum(f, 0)]
            nxt = np.where(xv <= self.threshold[idx], self.left[idx], self.right[idx])
            idx = np.where(at_leaf, idx, nxt)
        return self.leaf_proba[idx].mean(axis=0)


def export_forest(clf, path):
    """把 sklearn RandomForestClassifier 导出为 NumpyForest 可加载的 .npz。"""
    feature, threshold, left, right, proba, roots = [], [], [], [], [], []
    offset = 0
    for est in clf.estimators_:
        t = est.tree_
        roots.append(offset)
        feature.append(t.feature)
        threshold.append(t.threshold.astype(np.float32))
        l = t.children_left.copy()
        r = t.children_right.copy()
        l[l >= 0] += offset
        r[r >= 0] += offset
        left.append(l)
        right.append(r)
        v = t.value[:, 0, :].astype(np.float32)
        v /= np.maximum(v.sum(axis=1, keepdims=True), 1e-12)
        proba.append(v)
        offset += t.node_count

    # 用非压缩格式：文件大几 MB，但省去解压，冷加载再快一个量级
    np.savez(
        path,
        feature=np.concatenate(feature).astype(np.int32),
        threshold=np.concatenate(threshold),
        left=np.concatenate(left).astype(np.int32),
        right=np.concatenate(right).astype(np.int32),
        leaf_proba=np.vstack(proba),
        roots=np.array(roots, np.int32),
        classes=np.array([str(c) for c in clf.classes_]),
    )
