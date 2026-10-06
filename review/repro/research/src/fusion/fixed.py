"""保底固定融合（§5.1 保底模式）：CDF 校准后等权。

M0 默认；M2 学习路由必须通过门控（赢 margin 才替换）才接管。
"""
import numpy as np


def fuse(slot_scores_calibrated, weights=None):
    """slot_scores_calibrated: dict[str, float] -> 融合分数"""
    names = sorted(slot_scores_calibrated)
    if weights is None:
        weights = {n: 1.0 / len(names) for n in names}
    return float(sum(weights.get(n, 0.0) * slot_scores_calibrated[n] for n in names))
