"""共识自适应融合（§5.3，U14）：无标签槽位权重，与保底门控并存

demo3 的共识自适应在 zero-shot 场景实测有效（0.809→0.813），机制：
  各槽位与"槽位中位数"的 Pearson 相关性 relu^2 → 共识权重
  与等权 blend（blend=0.5，保守），再用保底门控决定是否启用

与 M2 路由的本质区别（§5.1 表格第 3 行）：
  - 路由：用 init_defect(30) 有标签训练 → U15 证实不可外推
  - 共识：完全无标签，只用 train/good 校准分 + 流数据 → 不受标签样本量限制

U14 落定：引擎以共识为在线自适应的唯一权重来源；路由保留为追溯通道。
"""
import numpy as np
from ..fusion.fixed import fuse


def consensus_weights(calibrated_scores, blend=0.5, hard_thresh=0.0):
    """calibrated_scores: dict[slot_name -> list[float]]（train/good 校准分）
    返回 dict[slot_name -> weight]。

    机制（对齐 demo3）：
    1. Z = 各槽位校准分矩阵 (n_slots, N)
    2. med = 槽位中位数（共识信号）
    3. rho_i = Pearson(Z_i, med)，w_i = relu(rho)^2
    4. 归一化后与等权 blend（blend=0.5：一半信共识，一半保等权底）

    hard_thresh（M4 攻坚，无标签，仅用 train/good 校准分）：
    纯共识权重 w < hard_thresh 的槽位（与共识信号不一致的噪声槽位）→ 权重置 0
    再归一化后与等权 blend——等价于裁剪的连续化版本（U15：任何有标签门控不可靠，
    此处仍是无标签路径）。hard_thresh=0 即原版。
    """
    names = sorted(calibrated_scores)
    if not names:
        raise ValueError("cannot build consensus weights: no active slots")
    Z = np.stack([np.asarray(calibrated_scores[n], dtype=np.float64) for n in names])
    med = np.median(Z, axis=0)
    zm = med - med.mean()
    w = []
    for i, n in enumerate(names):
        zi = Z[i] - Z[i].mean()
        denom = np.sqrt((zi ** 2).sum() * (zm ** 2).sum()) + 1e-12
        rho = float((zi * zm).sum() / denom)
        w.append(max(rho, 0.0) ** 2 if np.isfinite(rho) else 0.0)  # 退化槽位(常数分)=0 权重
    tot = sum(w)
    if tot <= 1e-9:
        w = [1.0 / len(names)] * len(names)
    else:
        w = [x / tot for x in w]
    # 硬门控：弱共识槽位置零（噪声槽位排除，等价裁剪的连续化）
    if hard_thresh > 0:
        w = [0.0 if x < hard_thresh else x for x in w]
        tot = sum(w)
        if tot > 1e-9:
            w = [x / tot for x in w]
        else:
            w = [1.0 / len(names)] * len(names)
    # 与等权 blend（保守起步）
    base = 1.0 / len(names)
    return {n: (1 - blend) * base + blend * w[i] for i, n in enumerate(names)}


def consensus_gate(base_auroc, consensus_auroc, margin=0.005):
    """与路由门控同构（§5.1）：共识 AUC ≥ 保底 + margin 才启用"""
    return bool(consensus_auroc >= base_auroc + margin)
