"""CDF 校准器（红线：仅用 train/good 拟合）

每槽位独立：直方图 CDF 把原始分数映射到 [0,1] 正常分位数，
含义="比多少比例的正常训练图更异常"——替代 demo4 的 flip 补丁。
"""
import numpy as np


class CDFCalibrator:
    def __init__(self, n_bins=256):
        self.n_bins = n_bins
        self.edges = None

    def fit(self, normal_scores):
        s = np.sort(np.asarray(normal_scores, dtype=np.float64))
        # 分位数边界，保证单调
        self.edges = np.quantile(s, np.linspace(0, 1, self.n_bins + 1))
        self.edges[0] -= 1e-9
        return self

    def transform(self, scores):
        s = np.asarray(scores, dtype=np.float64)
        n = len(self.edges)
        # 主体：分位映射（保序），内部 [0,1]
        idx = np.searchsorted(self.edges, s, side="right") - 1
        q = np.clip(idx, 0, n - 1) / (n - 1)
        # U32（2026-08-16）：超上界线性外推——旧实现把 s>train-max 全部截为 1.0，
        # 缺陷样本大量平局（trad cal 0.61 vs raw 0.996；component raw 0.857
        # → cal 0.643），等权融合/排序丢失缺陷内区分度。
        hi, lo = self.edges[-1], self.edges[-2] if n >= 2 else self.edges[-1] - 1.0
        slope = 1.0 / max(hi - lo, 1e-12)
        beyond = s > hi
        q[beyond] = 1.0 + (s[beyond] - hi) * slope
        # 超下界：线性外推（低于 train-min → 更正常，可为负）
        lo0, lo1 = self.edges[0], self.edges[1] if n >= 2 else self.edges[0] + 1.0
        below = s < lo0
        q[below] = (s[below] - lo0) / max(lo1 - lo0, 1e-12)
        # 限幅防单槽爆值主导融合（保序：超过 cap 的样本仍单调递增，仅在 cap 处截断）
        return np.clip(q, -1.0, 2.0)

    def fit_transform(self, normal_scores):
        return self.fit(normal_scores).transform(normal_scores)


class TplCalibrator:
    """tpl 槽位专用归一化（2026-08-30，P0：L3 模板差分）。

    背景：tpl 槽位 raw diff 中，训练正常图在模板库中"自匹配" → 分数≈0，
    CDF 校准的正常分布退化（edges 全 0），任何分数（含正常 0）都被映射到
    高分位 1.0 → 正常图在融合中贡献高值，误报为异常。

    方案：用 train 正常分数 p95 作参考做线性归一化——正常≈0、缺陷>1，
    保留"diff 越大越异常"语义（与 CDF 的 [0,1] 尺度 + 外推兼容）。
    """
    def __init__(self, n_bins=256):
        self.edges = np.array([1.0])   # 兼容 CDF 的 rollback/快照接口：edges[0] 即 ref

    @property
    def ref(self):
        # ref 由 edges 派生：快照还原/回滚只写 edges 也能保持一致
        return float(self.edges[0])

    @ref.setter
    def ref(self, v):
        self.edges = np.array([float(v)])

    def fit(self, normal_scores):
        arr = np.asarray(normal_scores, dtype=np.float64)
        # 正常分布退化（全 0）时兜底：任何 >0 的 diff 都视为异常
        self.ref = float(np.quantile(arr, 0.95)) if len(arr) else 0.0
        if self.ref <= 1e-6:
            self.ref = 0.05
        self.edges = np.array([self.ref])
        return self

    def transform(self, scores):
        q = np.asarray(scores, dtype=np.float64) / self.ref
        return np.clip(q, -1.0, 2.0)

    def fit_transform(self, normal_scores):
        return self.fit(normal_scores).transform(normal_scores)
