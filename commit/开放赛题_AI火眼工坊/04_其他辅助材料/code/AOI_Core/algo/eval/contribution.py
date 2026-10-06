"""三层特征评价 + 贡献档案 API（§3.2，§9.2 交付物）

回答"哪些特征独立、真正起作用"：
1. 单变量层：逐槽位 × 逐缺陷类型 AUROC（缺陷类型切片——全局平均销毁信息）
2. 冗余层：槽位分数间相关聚类（回答"哪些独立"）
3. 消融层：leave-one-slot-out 对融合分数的影响（回答"真正起作用"）

输出贡献档案（JSON）：每组有没有用、该不该留，供 AOI_sys 直接可视化。
红线：仅消费 eval 侧（test）数据做评价报告，不参与 fit/权重计算。
"""
import os
import json
import numpy as np
from .metrics import image_metrics
from ..fusion.fixed import fuse


def defect_type_from_path(path):
    """从路径父目录名推断缺陷类型（data_local 结构 {split}/{defect_type}/）。"""
    parent = os.path.basename(os.path.dirname(str(path)))
    return parent if parent else "unknown"


def univariate_per_type(slot_scores, labels, types):
    """单变量层：逐槽位 × 逐缺陷类型 AUROC。types: 每样本的缺陷类型（good 样本标 None）"""
    slot_names = list(slot_scores)
    type_names = sorted({t for t in types if t})
    out = {"by_type": {}, "overall": {}}
    for t in type_names:
        idx = [i for i, tt in enumerate(types) if tt == t or not tt]  # 该类型正样本 + 全部 good
        lab = [labels[i] for i in idx]
        out["by_type"][t] = {n: image_metrics([lab[j] for j in range(len(idx))],
                                              [slot_scores[n][i] for i in idx])["auroc"]
                             for n in slot_names}
    for n in slot_names:
        out["overall"][n] = image_metrics(labels, slot_scores[n])["auroc"]
    return out


def redundancy_correlation(slot_scores):
    """冗余层：槽位分数 Spearman 相关矩阵（秩相关，尺度无关）"""
    from scipy.stats import spearmanr
    slot_names = list(slot_scores)
    M = np.stack([np.asarray(slot_scores[n], dtype=np.float64) for n in slot_names])
    corr = np.ones((len(slot_names), len(slot_names)))
    for i in range(len(slot_names)):
        for j in range(i + 1, len(slot_names)):
            c = spearmanr(M[i], M[j]).statistic
            corr[i, j] = corr[j, i] = float(c)
    return {"names": slot_names, "corr": corr.tolist()}


def ablation_fusion(slot_scores, labels, weights=None):
    """消融层：全量融合 vs leave-one-slot-out vs 单槽位（§3.2 第 3 层）"""
    slot_names = list(slot_scores)
    cal = {n: np.asarray(slot_scores[n], dtype=np.float64) for n in slot_names}
    if weights is None:
        weights = {n: 1.0 / len(slot_names) for n in slot_names}

    def _auroc(sel_names):
        fused = [fuse({n: float(cal[n][i]) for n in sel_names}, weights) for i in range(len(labels))]
        return image_metrics(labels, fused)["auroc"]

    out = {"full": _auroc(slot_names), "single": {}, "leave_one_out": {}}
    for n in slot_names:
        out["single"][n] = _auroc([n])
        out["leave_one_out"][n] = _auroc([m for m in slot_names if m != n])
    return out


def contribution_profile(slot_scores, labels, types, weights=None):
    """三层评价汇总 -> 贡献档案 dict（§3.2 输出契约）"""
    return {
        "univariate": univariate_per_type(slot_scores, labels, types),
        "redundancy": redundancy_correlation(slot_scores),
        "ablation": ablation_fusion(slot_scores, labels, weights),
    }


def save_profile(profile, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)
