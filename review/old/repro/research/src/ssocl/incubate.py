"""M3 AHL 孵育头（§7.6）：同类缺陷样例≥8 时孵育专属头，强化拦截。

- 训练：逻辑回归区分「样例 patch（正）」vs「正常锚定 coreset patch（负）」
  用 DINO 归一化 patch 特征（384 维），参数量 ~384，符合"独立头不影响主干"
- 输出：孵育头加分 = 正类概率 top-k 均值 → 独立拦截通道（不进槽位表/融合权重，
  规避 §4.4 硬约束一——它不是参与融合的槽位，是缺陷样例拦截的强化版）
- 新样例继续累积 → 重新孵育（旧头丢弃，样本更新换代）
"""
import time
import numpy as np

from .banks import _norm_patch


class IncubatedHead:
    """孵育头：sklearn 逻辑回归（无网络依赖，CPU 快）。"""

    def __init__(self, topk=8, boost=0.3, min_samples=8, l2=1.0):
        self.topk = topk
        self.boost = boost
        self.min_samples = min_samples
        self.l2 = l2
        self.clf = None
        self.trained_at = None
        self.n_pos = 0

    def _ready(self, defect_bank):
        return len(defect_bank.samples) >= self.min_samples

    def incubate(self, defect_bank, normal_core):
        """训练头。normal_core: (M,384) fp16 锚定 coreset patch。
        正样本 = 全部缺陷样例 patch（可能量大 → 抽样上限 2 万）；负样本 = coreset 抽样等量。"""
        from sklearn.linear_model import LogisticRegression
        pos = torch_cat([s[0] for s in defect_bank.samples]).float().numpy()
        neg = normal_core.float().numpy()
        rng = np.random.default_rng(0)
        cap = 20_000
        if len(pos) > cap:
            pos = pos[rng.choice(len(pos), cap, replace=False)]
        n_neg = min(len(neg), len(pos))
        neg = neg[rng.choice(len(neg), n_neg, replace=False)]
        X = np.concatenate([pos, neg])
        y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        clf = LogisticRegression(C=1.0 / self.l2, max_iter=500)
        clf.fit(X, y)
        self.clf = clf
        self.trained_at = time.time()
        self.n_pos = len(pos)
        return self

    def score(self, tile_feats):
        """-> 孵育加分 ∈ [0, boost]（头未训练返回 0）。"""
        if self.clf is None:
            return 0.0
        f = _norm_patch(tile_feats).float().numpy()
        if len(f) == 0:
            return 0.0
        p = self.clf.predict_proba(f)[:, 1]
        k = min(self.topk, len(p))
        return round(float(np.sort(p)[-k:].mean()) * self.boost, 5)


def torch_cat(parts):
    import torch
    return torch.cat(parts, dim=0)


class HeadManager:
    """孵育头生命周期：样例≥8 孵育；样例数明显增长（≥1.5x）重新孵育。"""

    def __init__(self, cfg=None):
        c = cfg or {}
        self.head = None
        self.last_sample_n = 0
        self.cfg = c
        self.log = []

    def maybe_incubate(self, defect_bank, normal_core):
        if len(defect_bank.samples) < (c := self.cfg.get("min_samples", 8)):
            return None
        if self.head is not None and \
           len(defect_bank.samples) < self.last_sample_n * 1.5:
            return self.head
        h = IncubatedHead(min_samples=c,
                          boost=self.cfg.get("boost", 0.3),
                          topk=self.cfg.get("topk", 8))
        try:
            h.incubate(defect_bank, normal_core)
        except Exception as e:                # 训练失败（如类别退化）→ 保持旧头
            self.log.append({"t": time.time(), "ok": False, "err": str(e)})
            return self.head
        self.head = h
        self.last_sample_n = len(defect_bank.samples)
        self.log.append({"t": time.time(), "ok": True, "n_pos": h.n_pos})
        return h

    def boost(self, tile_feats):
        return self.head.score(tile_feats) if self.head else 0.0
