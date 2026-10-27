"""保底固定融合（§5.1 保底模式）：CDF 校准后等权。

M0 默认；M2 学习路由必须通过门控（赢 margin 才替换）才接管。
"""
import numpy as np


def effective_weights(names, weights=None, confidences=None):
    """仅活动槽参与：w'_k = w_k*c_k / sum(w*c)。零置信停用，无有效权重报错。"""
    names = sorted(names)
    if not names:
        raise ValueError("cannot fuse scores: no active slots")
    result = {}
    for n in names:
        w = 1.0 if weights is None else float(weights.get(n, 0.0))
        c = 1.0 if confidences is None else float(confidences.get(n, 1.0))
        if not np.isfinite(w) or w < 0 or not np.isfinite(c) or not 0 <= c <= 1:
            raise ValueError("fusion weights/confidences must be finite and valid")
        result[n] = w * c
    total = sum(result.values())
    if total <= 0:
        raise ValueError("cannot fuse scores: no effective slot weight")
    return {n: w / total for n, w in result.items()}


def fuse(slot_scores_calibrated, weights=None, confidences=None):
    """校准后按活动槽置信加权；旧调用默认 c_k=1，停用槽不占分母。"""
    active = effective_weights(slot_scores_calibrated, weights, confidences)
    return float(sum(active[n] * slot_scores_calibrated[n] for n in active))
