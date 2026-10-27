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


def _metric(labels, scores):
    """Return JSON-safe metric values without turning undefined results into zero."""
    if len(np.unique(labels)) < 2:
        return None, {"status": "insufficient", "reason": "需要同时包含正常和缺陷样本"}
    result = image_metrics(labels, scores)
    value = result["auroc"]
    if not np.isfinite(value):
        return None, {"status": "insufficient", "reason": "AUROC 非有限，当前样本不足以计算"}
    return value, {"status": "available", "reason": None}


def defect_type_from_path(path):
    """从路径父目录名推断缺陷类型（data_local 结构 {split}/{defect_type}/）。"""
    parent = os.path.basename(os.path.dirname(str(path)))
    return parent if parent else "unknown"


def univariate_per_type(slot_scores, labels, types):
    """单变量层：逐槽位 × 逐缺陷类型 AUROC。types: 每样本的缺陷类型（good 样本标 None）"""
    slot_names = list(slot_scores)
    type_names = sorted({t for t in types if t})
    out = {"by_type": {}, "overall": {}, "availability": {"by_type": {}, "overall": {}}}
    for t in type_names:
        idx = [i for i, tt in enumerate(types) if tt == t or not tt]  # 该类型正样本 + 全部 good
        lab = [labels[i] for i in idx]
        out["by_type"][t] = {}
        out["availability"]["by_type"][t] = {}
        for n in slot_names:
            value, status = _metric(lab, [slot_scores[n][i] for i in idx])
            out["by_type"][t][n] = value
            out["availability"]["by_type"][t][n] = status
    for n in slot_names:
        value, status = _metric(labels, slot_scores[n])
        out["overall"][n] = value
        out["availability"]["overall"][n] = status
    return out


def redundancy_correlation(slot_scores):
    """冗余层：槽位分数 Spearman 相关矩阵（秩相关，尺度无关）"""
    from scipy.stats import spearmanr
    slot_names = list(slot_scores)
    M = np.stack([np.asarray(slot_scores[n], dtype=np.float64) for n in slot_names])
    corr = np.ones((len(slot_names), len(slot_names)), dtype=object)
    availability = [[{"status": "available", "reason": None}
                     for _ in slot_names] for _ in slot_names]
    for i in range(len(slot_names)):
        for j in range(i + 1, len(slot_names)):
            c = spearmanr(M[i], M[j]).statistic
            if not np.isfinite(c):
                corr[i, j] = corr[j, i] = None
                status = {"status": "insufficient", "reason": "槽位分数无变化，无法计算秩相关"}
                availability[i][j] = availability[j][i] = status
            else:
                corr[i, j] = corr[j, i] = float(c)
    return {"names": slot_names, "corr": corr.tolist(),
            "availability": availability}


def ablation_fusion(slot_scores, labels, weights=None):
    """消融层：全量融合 vs leave-one-slot-out vs 单槽位（§3.2 第 3 层）"""
    slot_names = list(slot_scores)
    cal = {n: np.asarray(slot_scores[n], dtype=np.float64) for n in slot_names}
    if weights is None:
        weights = {n: 1.0 / len(slot_names) for n in slot_names}

    def _auroc(sel_names):
        fused = [fuse({n: float(cal[n][i]) for n in sel_names}, weights) for i in range(len(labels))]
        return _metric(labels, fused)

    full, full_status = _auroc(slot_names)
    out = {"full": full, "single": {}, "leave_one_out": {},
           "availability": {"full": full_status, "single": {}, "leave_one_out": {}}}
    for n in slot_names:
        value, status = _auroc([n])
        out["single"][n] = value
        out["availability"]["single"][n] = status
        value, status = _auroc([m for m in slot_names if m != n])
        out["leave_one_out"][n] = value
        out["availability"]["leave_one_out"][n] = status
    return out


def contribution_profile(slot_scores, labels, types, weights=None):
    """三层评价汇总 -> 贡献档案 dict（§3.2 输出契约）"""
    return {
        "univariate": univariate_per_type(slot_scores, labels, types),
        "redundancy": redundancy_correlation(slot_scores),
        "ablation": ablation_fusion(slot_scores, labels, weights),
    }


def _auc(labels, scores):
    v, _ = _metric(labels, scores)
    return v


def _boot_ci(labels, scores, n_boot=200, seed=0):
    """分层 bootstrap 95% CI；任一重采样缺类即跳过。"""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    pos, neg = np.where(labels == 1)[0], np.where(labels == 0)[0]
    if len(pos) < 2 or len(neg) < 2:
        return None
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        a = _auc(labels[idx], scores[idx])
        if a is not None:
            vals.append(a)
    if len(vals) < 10:
        return None
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def _knn_skip1(d):
    """(M,N) 距离 -> 与 TradSlot.score_image 同口径：跳最近邻取 [1:4] 均值。"""
    s = np.sort(d, axis=1)
    return s[:, 1:4].mean(axis=1) if s.shape[1] > 1 else s[:, 0]


def _clusters(names, corr, thr):
    """相关簇（|spearman|>thr 并查集）。"""
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i, a in enumerate(names):
        for j in range(i + 1, len(names)):
            c = corr[i][j]
            if c is not None and abs(c) > thr:
                parent[find(a)] = find(names[j])
    out = {}
    for n in names:
        out.setdefault(find(n), []).append(n)
    return [sorted(v) for v in out.values()]


def trad_group_profile(trad, imgs, labels, types, category=None, data_source="test",
                       config=None, n_boot=200, cluster_thr=0.8, seed=0):
    """AC-012 trad 组级贡献档案（版本×组×品类×缺陷×视图）。

    每组：单组 AUROC+CI+方向、固定留一（总距离去掉该组项，近邻不重算）、
    重建留一（剩余组上重算 kNN）、排列重要性（eval 内打乱该组维）、
    相关簇留一、冗余（与其他组最大 |spearman|）、提取耗时、样本数。
    红线：仅评价报告，不回写 fit/权重/选组。
    """
    from scipy.stats import spearmanr
    from ..features import registry as freg
    import time as _time
    labels = [int(y) for y in labels]
    meta = {"registry_version": freg.REGISTRY_VERSION, "registry_sha256": freg.registry_hash(),
            "signature": trad.signature() if hasattr(trad, "signature") else None,
            "category": category, "data_source": data_source, "config": config,
            "n_normal": int(sum(1 for y in labels if y == 0)),
            "n_defect": int(sum(1 for y in labels if y == 1)),
            "views": {"global": "available",
                      "tile": "not_applicable：trad 仅整图库打分，局部视图由 DINO 局部分支负责"},
            "created_at": _time.strftime("%Y-%m-%d %H:%M:%S"),
            "semantics": "组距离=偏离正常程度；AUROC 为开发/评价集判别力，非缺陷概率，非融合贡献"}
    if getattr(trad, "bank_w", None) is None or not imgs:
        return {"status": "unavailable", "reason": "trad 整图库未建立或无评价图", "meta": meta}
    if len(set(labels)) < 2:
        return {"status": "insufficient", "reason": "评价集需同时包含正常和缺陷", "meta": meta}
    # 提取 eval z 向量 + 逐组耗时
    t_acc = {}
    zs = []
    for im in imgs:
        zs.append(trad._z(trad._vec(im), trad.norm_w))
        for key, _end, ms in getattr(trad.ext, "last_trace", None) or []:
            t_acc.setdefault(key, []).append(float(ms))
    Z = np.stack(zs).astype(np.float64)
    B = trad.bank_w.astype(np.float64)
    dim = B.shape[1]
    grouped = trad.scoring == "grouped" and bool(trad.group_weight)
    # legacy 与生产同口径用组全维；grouped 用剔除近零方差后的维
    gdims = {}
    for k in freg.GROUP_KEYS:
        ids = (trad._group_dims(dim, [k]).get(k) if grouped
               else np.asarray([i for i in freg.dim_index([k], dim)]))
        if ids is not None and len(ids):
            gdims[k] = ids
    keys = list(gdims)

    # 每组平方距离 (M,N)；legacy 组项=平方 L2，grouped 组项=w*L2/scale
    sq, term = {}, {}
    for k in keys:
        ids = gdims[k]
        zb, bb = Z[:, ids], B[:, ids]
        d2 = np.maximum((zb ** 2).sum(1)[:, None] + (bb ** 2).sum(1)[None, :] - 2 * zb @ bb.T, 0.0)
        sq[k] = d2
        if grouped:
            term[k] = trad.group_weight.get(k, 0.0) * np.sqrt(d2) / trad.group_scale.get(k, 1.0)
        else:
            term[k] = d2
    enabled = [k for k in keys if k in trad.groups and (not grouped or trad.group_weight.get(k, 0) > 0)]

    def total(sel):
        if not sel:
            return None
        acc = sum(term[k] for k in sel)
        return acc if grouped else np.sqrt(acc)

    full_d = total(enabled)
    full_scores = _knn_skip1(full_d)
    full_auc = _auc(labels, full_scores)
    order = np.argsort(full_d, axis=1)[:, 1:4] if full_d.shape[1] > 1 else np.argsort(full_d, axis=1)[:, :1]

    single = {k: _knn_skip1(np.sqrt(sq[k])) for k in keys}
    names = list(keys)
    corr = [[1.0] * len(names) for _ in names]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            c = spearmanr(single[names[i]], single[names[j]]).statistic
            corr[i][j] = corr[j][i] = float(c) if np.isfinite(c) else None
    clusters = _clusters(enabled, [[corr[names.index(a)][names.index(b)] for b in enabled]
                                   for a in enabled], cluster_thr) if enabled else []

    rng = np.random.RandomState(seed)
    groups = {}
    per_image = []
    by_type = sorted({t for t in types if t})
    for k in keys:
        g = freg.group(k)
        a = _auc(labels, single[k])
        rec = {"name": g["name"], "category": g["category"], "version": g["version"],
               "dim": int(len(gdims[k])), "enabled": k in enabled,
               "weight": float(trad.group_weight.get(k, 1.0 if k in enabled else 0.0)),
               "single_auroc": a, "single_ci95": _boot_ci(labels, single[k], n_boot, seed),
               "direction": None if a is None else ("reversed" if a < 0.5 else "normal"),
               "extract_ms": float(np.mean(t_acc[k])) if k in t_acc else None,
               "by_type": {}}
        for t in by_type:
            idx = [i for i, tt in enumerate(types) if tt == t or not tt]
            rec["by_type"][t] = _auc([labels[i] for i in idx], single[k][idx])
        if k in enabled and len(enabled) > 1:
            rest = [x for x in enabled if x != k]
            # 固定留一：近邻索引不变，仅从总距离中去掉该组项
            fixed_d = total(rest)
            fixed = np.take_along_axis(fixed_d, order, axis=1).mean(axis=1)
            rec["fixed_loo_auroc"] = _auc(labels, fixed)
            ablated_scores = _knn_skip1(fixed_d)
            rec["rebuild_loo_auroc"] = _auc(labels, ablated_scores)
            for i, (baseline, ablated) in enumerate(zip(full_scores, ablated_scores)):
                per_image.append({"image_index": i, "feature_group": k,
                                  "baseline": float(baseline), "ablated": float(ablated),
                                  "contribution": float(baseline - ablated)})
            # 排列：eval 内打乱该组维（破坏样本-特征对应）
            perm = rng.permutation(len(Z))
            t_perm = dict(term)
            ids = gdims[k]
            zp = Z[perm][:, ids]
            bb = B[:, ids]
            d2p = np.maximum((zp ** 2).sum(1)[:, None] + (bb ** 2).sum(1)[None, :] - 2 * zp @ bb.T, 0.0)
            t_perm[k] = (trad.group_weight.get(k, 0.0) * np.sqrt(d2p) / trad.group_scale.get(k, 1.0)
                         if grouped else d2p)
            acc = sum(t_perm[x] for x in enabled)
            pa = _auc(labels, _knn_skip1(acc if grouped else np.sqrt(acc)))
            rec["permutation_auroc"] = pa
            for f in ("fixed_loo", "rebuild_loo", "permutation"):
                v = rec[f + "_auroc"]
                rec[f + "_drop"] = None if v is None or full_auc is None else float(full_auc - v)
        else:
            for f in ("fixed_loo", "rebuild_loo", "permutation"):
                rec[f + "_auroc"] = rec[f + "_drop"] = None
            rec["loo_status"] = "not_enabled" if k not in enabled else "only_group"
        i = names.index(k)
        others = [(names[j], corr[i][j]) for j in range(len(names)) if j != i and corr[i][j] is not None]
        rec["redundancy"] = max(others, key=lambda x: abs(x[1]))[0:2] if others else None
        groups[k] = rec
    cluster_loo = []
    for cl in clusters:
        rest = [x for x in enabled if x not in cl]
        v = _auc(labels, _knn_skip1(total(rest))) if rest else None
        cluster_loo.append({"groups": cl, "loo_auroc": v,
                            "drop": None if v is None or full_auc is None else float(full_auc - v)})
    return {"status": "available", "meta": meta, "view": "global",
            "full_auroc": full_auc, "full_ci95": _boot_ci(labels, full_scores, n_boot, seed),
            "enabled": enabled, "groups": groups,
            "redundancy": {"names": names, "corr": corr, "cluster_thr": cluster_thr},
            "cluster_loo": cluster_loo,
            "per_image": per_image,
            "per_image_status": "available" if per_image else "unavailable",
            "per_image_reason": None if per_image else "至少需要两个已启用的有效特征组才能留一",
            "per_image_semantics": "传统特征原始距离；重算近邻留一；contribution=baseline-ablated，非融合分数或因果贡献"}


def save_profile(profile, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)
